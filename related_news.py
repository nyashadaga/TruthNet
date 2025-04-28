import streamlit as st
import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Set page config
st.set_page_config(
    page_title="📰 Related News Finder",
    page_icon="📰",
    layout="wide",
)

# Load environment variables
load_dotenv()

# Download NLTK resources if needed
try:
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
except:
    sia = None

# API URLs
NEWS_API_URL = "https://newsapi.org/v2/everything"
GNEWS_URL = "https://gnews.io/api/v4/search"
MEDIASTACK_URL = "http://api.mediastack.com/v1/news"
NEWSDATA_URL = "https://newsdata.io/api/1/news"

# NewsAPI Configuration
NEWS_API_KEYS = [
    os.getenv("NEWS_API_KEY_1"),
    os.getenv("NEWS_API_KEY_2"),
    os.getenv("NEWS_API_KEY_3"),
    os.getenv("NEWS_API_KEY_4"),
]
NEWS_API_KEYS = [key for key in NEWS_API_KEYS if key]
if not NEWS_API_KEYS:
    st.error("Please set at least one NEWS_API_KEY in your environment variables")
    st.stop()
current_api_key_index = 0

def get_next_api_key():
    global current_api_key_index
    key = NEWS_API_KEYS[current_api_key_index]
    current_api_key_index = (current_api_key_index + 1) % len(NEWS_API_KEYS)
    return key

# Load spaCy model once
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    st.warning("spaCy model not found. Falling back to basic keyword extraction.")

# Improved headline processing for better keyword extraction
def process_headline(headline):
    if nlp:
        doc = nlp(headline.lower())
        
        # Extract named entities first (highest priority)
        entities = [ent.text for ent in doc.ents]
        
        # Extract key tokens with weighted importance
        word_freq = {}
        for token in doc:
            if token.is_alpha and not token.is_stop and len(token.text) > 2:
                word = token.lemma_
                # Weight by part of speech
                importance = 1.0
                if token.pos_ == "PROPN":  # Proper nouns get highest weight
                    importance = 3.0
                elif token.pos_ == "NOUN":
                    importance = 2.0
                elif token.pos_ == "VERB":
                    importance = 1.5
                
                word_freq[word] = word_freq.get(word, 0) + importance
        
        # Get top keywords
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        keywords = [word for word, _ in keywords]
        
        # Combine entities and keywords
        search_terms = entities + [k for k in keywords if k not in entities]
        search_terms = search_terms[:7]  # Limit to top 7 terms
        
        # Create search query with exact headline and expanded terms
        search_query = f'"{headline}" OR ' + " OR ".join(search_terms)
        
        # Add entity pairs for better context
        if len(entities) >= 2:
            for i in range(len(entities)-1):
                search_query += f' OR "{entities[i]} {entities[i+1]}"'
        
        return search_query, search_terms
    else:
        # Fallback method
        words = headline.lower().split()
        words = [word for word in words if len(word) > 3]
        return " OR ".join(words[:3]), words[:3]

def parse_date(date_str):
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return datetime.min

# New function to assess article credibility
def assess_article_credibility(article):
    """Calculate a credibility score for an article based on multiple factors"""
    score = 60  # Start with neutral score
    
    # Check source reputation
    known_credible_sources = {
        "Reuters": 95, "Associated Press": 95, "BBC News": 90, 
        "The New York Times": 85, "The Washington Post": 85, 
        "The Guardian": 85, "Bloomberg": 85, "CNN": 80, "NPR": 85,
        "The Wall Street Journal": 85, "The Economist": 85
    }
    source = article.get("source", "Unknown")
    if source in known_credible_sources:
        score += known_credible_sources[source] * 0.3
    
    # Analyze sentiment extremity (extreme sentiment might indicate bias)
    if sia:
        text = article.get("title", "") + " " + article.get("description", "")
        sentiment = sia.polarity_scores(text)
        compound = abs(sentiment['compound'])
        
        # More neutral sentiment gets higher credibility
        if compound > 0.8:
            score -= 15  # Very extreme sentiment
        elif compound > 0.6:
            score -= 5
        elif compound < 0.2:
            score += 10  # Balanced reporting
    
    # Check for clickbait patterns
    title = article.get("title", "").lower()
    clickbait_patterns = ["you won't believe", "shocking", "mind blowing", 
                         "this is why", "?!", "!!!", "secret", "revealed"]
    for pattern in clickbait_patterns:
        if pattern in title:
            score -= 10
            break
    
    # Assess content length and quality
    description = article.get("description", "")
    if description:
        if len(description) < 50:
            score -= 10  # Very short descriptions are suspicious
        elif len(description) > 200:
            score += 5   # Longer descriptions suggest more substance
            
    # Ensure score stays within 0-100 range
    return max(0, min(100, score))

# Calculate relevance score for articles
def calculate_relevance(article, search_terms):
    """Calculate how relevant an article is to the search terms"""
    # Combine available text
    article_text = ""
    if article.get("title"):
        article_text += article["title"] + " "
    if article.get("description"):
        article_text += article["description"] + " "
    if article.get("content"):
        article_text += article.get("content", "") + " "
    
    article_text = article_text.lower()
    
    # Count term occurrences and exact matches
    term_count = 0
    exact_match_bonus = 0
    
    for term in search_terms:
        term = term.lower()
        term_count += article_text.count(term)
        
        if term in article_text:
            exact_match_bonus += 1
    
    # Calculate term proximity (terms appearing close together)
    proximity_score = 0
    for i in range(len(search_terms)):
        for j in range(i+1, len(search_terms)):
            term1 = search_terms[i].lower()
            term2 = search_terms[j].lower()
            if term1 in article_text and term2 in article_text:
                # Find all occurrences of both terms
                idx1 = [i for i in range(len(article_text)) if article_text.startswith(term1, i)]
                idx2 = [i for i in range(len(article_text)) if article_text.startswith(term2, i)]
                
                # Calculate minimum distance between terms
                if idx1 and idx2:
                    min_dist = min(abs(i1 - i2) for i1 in idx1 for i2 in idx2)
                    proximity_score += max(0, 100 - min_dist) / 100
    
    # Calculate final relevance (higher is better)
    base_score = term_count * 10 + exact_match_bonus * 15 + proximity_score * 25
    
    # Normalize to 0-100 scale
    return min(100, base_score)

# Get articles from NewsAPI with relevance scoring
def get_newsapi_articles(query, search_terms):
    api_key = get_next_api_key()
    params = {
        "q": query,
        "apiKey": api_key,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 15,  # Get more articles for better selection
        "searchIn": "title,description,content",  # Search more fields
    }
    try:
        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        
        processed_articles = []
        for article in articles:
            relevance_score = calculate_relevance(article, search_terms)
            credibility_score = assess_article_credibility(article)
            
            processed_articles.append({
                "title": article["title"],
                "url": article["url"],
                "source": article.get("source", {}).get("name", "Unknown"),
                "date": article.get("publishedAt", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "relevance": relevance_score,
                "credibility": credibility_score,
                "api_source": "NewsAPI"
            })
        return processed_articles
    except requests.RequestException as e:
        st.warning(f"NewsAPI error: {e}")
        return []

# Similar updates for other API functions
def get_gnews_articles(query, search_terms):
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        return []
    params = {
        "q": query,
        "token": api_key,
        "lang": "en",
        "max": 15,
        "sortby": "relevance",
    }
    try:
        response = requests.get(GNEWS_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        
        processed_articles = []
        for article in articles:
            relevance_score = calculate_relevance(article, search_terms)
            credibility_score = assess_article_credibility(article)
            
            processed_articles.append({
                "title": article["title"],
                "url": article["url"],
                "source": article.get("source", {}).get("name", "Unknown"),
                "date": article.get("publishedAt", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "relevance": relevance_score,
                "credibility": credibility_score,
                "api_source": "GNews"
            })
        return processed_articles
    except requests.RequestException as e:
        st.warning(f"GNews error: {e}")
        return []

def get_mediastack_articles(query, search_terms):
    api_key = os.getenv("MEDIASTACK_API_KEY")
    if not api_key:
        return []
    params = {
        "access_key": api_key,
        "keywords": query,
        "languages": "en",
        "limit": 15,
        "sort": "published_desc",
    }
    try:
        response = requests.get(MEDIASTACK_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("data", [])
        
        processed_articles = []
        for article in articles:
            relevance_score = calculate_relevance(article, search_terms)
            credibility_score = assess_article_credibility(article)
            
            processed_articles.append({
                "title": article["title"],
                "url": article["url"],
                "source": article.get("source", "Unknown"),
                "date": article.get("published_at", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "relevance": relevance_score,
                "credibility": credibility_score,
                "api_source": "MediaStack"
            })
        return processed_articles
    except requests.RequestException as e:
        st.warning(f"MediaStack error: {e}")
        return []

def get_newsdata_articles(query, search_terms):
    api_key = os.getenv("NEWSDATA_API_KEY")
    if not api_key:
        return []
    params = {
        "apikey": api_key,
        "q": query,
        "language": "en",
        "size": 15
    }
    try:
        response = requests.get(NEWSDATA_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("results", [])
        
        processed_articles = []
        for article in articles:
            relevance_score = calculate_relevance(article, search_terms)
            credibility_score = assess_article_credibility(article)
            
            processed_articles.append({
                "title": article["title"],
                "url": article["link"],
                "source": article.get("source_id", "Unknown"),
                "date": article.get("pubDate", ""),
                "description": article.get("description", ""),
                "content": article.get("content", ""),
                "relevance": relevance_score,
                "credibility": credibility_score,
                "api_source": "NewsData"
            })
        return processed_articles
    except requests.RequestException as e:
        st.warning(f"NewsData.io error: {e}")
        return []

# Predict news veracity based on related articles
def predict_news_veracity(articles):
    """Determine the likelihood that the news is true"""
    if not articles:
        return 30  # Default to low-moderate confidence if no articles
    
    # Calculate various trustworthiness indicators
    
    # 1. Average credibility of top relevant articles
    top_articles = sorted(articles, key=lambda x: x["relevance"], reverse=True)[:5]
    top_credibility = [a["credibility"] for a in top_articles]
    avg_credibility = sum(top_credibility) / len(top_credibility) if top_credibility else 50
    
    # 2. Source diversity (more diverse sources = more credible)
    unique_sources = len(set(a["source"] for a in articles))
    source_diversity = min(100, unique_sources * 10)
    
    # 3. Consistency of reporting (lower standard deviation = more consistent)
    credibility_scores = [a["credibility"] for a in articles]
    consistency = 100 - (np.std(credibility_scores) * 5) if len(credibility_scores) > 1 else 50
    consistency = max(0, min(100, consistency))
    
    # 4. Balance of high-credibility sources
    high_credibility_count = sum(1 for a in articles if a["credibility"] > 75)
    high_cred_ratio = min(100, (high_credibility_count / len(articles)) * 100)
    
    # 5. Average relevance scores (higher relevance = stronger signal)
    avg_relevance = sum(a["relevance"] for a in articles) / len(articles)
    relevance_factor = avg_relevance / 100  # Scale from 0-1
    
    # Calculate weighted prediction score
    weights = {
        "avg_credibility": 0.4,
        "source_diversity": 0.15,
        "consistency": 0.2,
        "high_cred_ratio": 0.25
    }
    
    # Base prediction score
    prediction = (
        avg_credibility * weights["avg_credibility"] +
        source_diversity * weights["source_diversity"] +
        consistency * weights["consistency"] +
        high_cred_ratio * weights["high_cred_ratio"]
    )
    
    # Adjust by relevance factor (higher relevance should boost confidence)
    prediction = prediction * (0.7 + 0.3 * relevance_factor)
    
    return max(0, min(100, prediction))

# Main function to get related news with improved prediction
def get_related_news(headline):
    search_query, search_terms = process_headline(headline)
    all_articles = []
    
    # Collect articles from all sources
    all_articles.extend(get_newsapi_articles(search_query, search_terms))
    all_articles.extend(get_gnews_articles(search_query, search_terms))
    all_articles.extend(get_mediastack_articles(search_query, search_terms))
    all_articles.extend(get_newsdata_articles(search_query, search_terms))
    
    # Remove duplicates while preserving higher relevance scores
    seen_urls = {}
    for article in all_articles:
        url = article["url"]
        if url not in seen_urls or article["relevance"] > seen_urls[url]["relevance"]:
            seen_urls[url] = article
    
    unique_articles = list(seen_urls.values())
    
    # Calculate prediction metrics
    model_prediction = predict_news_veracity(unique_articles)
    
    # Calculate source credibility (separate from model prediction)
    top_articles = sorted(unique_articles, key=lambda x: x["relevance"], reverse=True)[:10]
    source_credibility = sum(a["credibility"] for a in top_articles) / len(top_articles) if top_articles else 60
    
    # Overall confidence combines model prediction and source credibility
    overall_confidence = (model_prediction * 0.6 + source_credibility * 0.4)
    
    # Sort by relevance first, then by date
    unique_articles.sort(key=lambda x: (-x["relevance"], -parse_date(x["date"]).timestamp()))
    
    return unique_articles, overall_confidence, model_prediction, source_credibility

def main():
    st.title("📰 Related News Finder")
    st.markdown("Enter a news headline to find related articles from various sources.")
    headline = st.text_input(
        "Enter a news headline:",
        placeholder="e.g., 'President announces new economic policy'",
    )
    if headline:
        with st.spinner("Searching for related articles..."):
            articles, overall_confidence, model_prediction, source_credibility = get_related_news(headline)
            
            # Display confidence metrics with more information
            col1, col2, col3 = st.columns(3)
            col1.metric("Overall Confidence", f"{overall_confidence:.0f}%")
            col2.metric("Model Prediction", f"{model_prediction:.0f}%")
            col3.metric("Source Credibility", f"{source_credibility:.0f}%")
            
            # Show breakdown of metrics for transparency
            with st.expander("Confidence Breakdown"):
                st.markdown("""
                - **Model Prediction**: Evaluates the likelihood that this is true news based on 
                  source diversity, reporting consistency, and credibility patterns
                - **Source Credibility**: Assesses the reputation of news sources and their reporting quality
                - **Overall Confidence**: Weighted combination of both metrics (60% model, 40% sources)
                """)
            
            if articles:
                st.subheader("Related News Articles")
                for article in articles:
                    with st.expander(f"{article['title']} ({article['relevance']:.0f}% Relevance)"):
                        st.markdown(f"**Source:** {article['source']} ({article['credibility']:.0f}% Credibility)")
                        st.markdown(f"**Date:** {article['date']}")
                        st.markdown(f"**Description:** {article['description']}")
                        st.markdown(f"**API Source:** {article['api_source']}")
                        st.markdown(f"[Read more]({article['url']})")
            else:
                st.info("No related articles found.")

if __name__ == "__main__":
    main()
