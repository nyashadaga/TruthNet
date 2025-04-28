import streamlit as st
import numpy as np
from scrape import ArticleScraper, c
import pickle
import math
import re
import string
import requests
from urllib.parse import urlparse
import os
from dotenv import load_dotenv
from gnews import GNews
from newsapi import NewsApiClient
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

st.title("Truth Net")

# Constants 
MAX_SEQ_LENGTH = 200  

# Model Loading Section 
@st.cache_resource
def load_models():
    try:
        # Load all required components
        pa_model = pickle.load(open(r"C:\Users\nyash\FAKE_NEWS\Copy\pickles\PA.pickle", "rb"))
        vectorizer = pickle.load(open(r"C:\Users\nyash\FAKE_NEWS\Copy\pickles\tfidf_vectorizer.pickle", "rb"))
        ensemble_model = pickle.load(open(r"C:\Users\nyash\FAKE_NEWS\Copy\pickles\ensemble.pickle", "rb"))
        
        # Load BiLSTM model and tokenizer
        bilstm_model = load_model(r"C:\Users\nyash\FAKE_NEWS\Copy\pickles\bilstm_model.keras")
        tokenizer = pickle.load(open(r"C:\Users\nyash\FAKE_NEWS\Copy\pickles\tokenizer.pickle", "rb"))
        
        return pa_model, vectorizer, ensemble_model, bilstm_model, tokenizer
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None, None

pa_model, vectorizer, ensemble_model, bilstm_model, tokenizer = load_models()

# Check all models are loaded properly
if not all([pa_model, vectorizer, ensemble_model, bilstm_model, tokenizer]):
    st.stop()

# Helper Functions
@st.cache_data
def check_robots_txt(url):
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"
        response = requests.get(robots_url, timeout=3)
        if response.status_code != 200:
            return True
        rules = response.text.split("\n")
        current_user_agent = None
        disallowed_paths = []
        for rule in rules:
            rule = rule.strip()
            if not rule or rule.startswith("#"):
                continue
            if rule.lower().startswith("user-agent:"):
                current_user_agent = rule.split(":", 1)[1].strip()
            elif current_user_agent == "*" and rule.lower().startswith("disallow:"):
                path = rule.split(":", 1)[1].strip()
                disallowed_paths.append(path)
        target_path = parsed_url.path
        for disallowed in disallowed_paths:
            if disallowed and target_path.startswith(disallowed):
                st.error("❌ Scraping is not allowed for this URL")
                return False
        st.success("✅ Scraping is allowed for this URL")
        return True
    except Exception as e:
        return True

@st.cache_data
def wordopt(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text

def hybrid_predict(content):
    # Preprocess for traditional models
    test_x = wordopt(content)
    tfidf_x = vectorizer.transform([test_x])
    
    # Get PA model prediction
    pa_pred = pa_model.predict(tfidf_x)[0]
    
    # Get ensemble prediction probability
    ensemble_prob = ensemble_model.predict_proba(tfidf_x)[0][1]
    
    # Preprocess for BiLSTM
    sequence = tokenizer.texts_to_sequences([test_x])
    padded = pad_sequences(sequence, maxlen=MAX_SEQ_LENGTH)
    bilstm_prob = bilstm_model.predict(padded, verbose=0)[0][0]
    
    # Weighted combination
    final_score = (0.3 * pa_pred) + (0.4 * ensemble_prob) + (0.3 * bilstm_prob)
    return final_score * 100  # Convert to percentage

# Credibility Scoring 
@st.cache_data
def assess_source_credibility(url, content, title=None):
    score = 50
    factors = []
    parsed = urlparse(url)
    domain = parsed.netloc.lower()

    # 1. HTTPS
    if parsed.scheme == "https":
        score += 10
        factors.append("✅ Uses HTTPS (secure connection)")
    else:
        score -= 10
        factors.append("❌ Does not use HTTPS")

    # 2. Domain reputation
    trusted_domains = [
    # International/English-language
    "bbc.com", "nytimes.com", "reuters.com", "apnews.com", "theguardian.com",
    "npr.org", "aljazeera.com", "washingtonpost.com", "cnn.com", 

    # Major Indian English-language news
    "timesofindia.indiatimes.com", "indiatimes.com", "ndtv.com", "indiatoday.in",
    "indianexpress.com", "thehindu.com", "news18.com", "firstpost.com",
    "hindustantimes.com", "deccanherald.com", "business-standard.com", "theprint.in",
    "livemint.com", "theweek.in", "tribuneindia.com", "telegraphindia.com", "rediff.com",

    # Indian Hindi/Regional news
    "aajtak.in", "abplive.com", "amarujala.com", "livehindustan.com", "jagran.com",
    "patrika.com", "bhaskar.com", "webdunia.com", "navbharattimes.indiatimes.com",

    # Indian news agencies and broadcasters
    "ddnews.gov.in", "allindiaradio.gov.in", "ptinews.com", "uniindia.com", "ani.in",
    "ians.in", "zee.com", "zeenews.india.com", "etvbharat.com", "cnnnews18.com",

    # Others (regional, business, fact-check, etc.)
    "businessinsider.in", "scroll.in", "newsbytesapp.com", "factchecker.in", "boomlive.in"
]

    if any(td in domain for td in trusted_domains):
        score += 20
        factors.append(f"✅ Trusted news domain: {domain}")
    else:
        score -= 5
        factors.append(f"ℹ️ Domain not in trusted list: {domain}")

    # 3. About/Contact page presence
    about_url = f"{parsed.scheme}://{parsed.netloc}/about"
    contact_url = f"{parsed.scheme}://{parsed.netloc}/contact"
    try:
        about_resp = requests.get(about_url, timeout=3)
        contact_resp = requests.get(contact_url, timeout=3)
        if about_resp.status_code == 200 or contact_resp.status_code == 200:
            score += 10
            factors.append("✅ About/Contact page found (transparency)")
        else:
            score -= 5
            factors.append("ℹ️ No About/Contact page found")
    except:
        factors.append("ℹ️ Could not check About/Contact page")

    # 4. Recency
    this_year = str(datetime.datetime.now().year)
    if this_year in content:
        score += 5
        factors.append("✅ Article mentions current year (recent)")
    else:
        score -= 2
        factors.append("ℹ️ Article may not be recent")

    # 5. Headline-content match
    if title:
        title_words = set(title.lower().split())
        content_words = set(content.lower().split())
        overlap = len(title_words & content_words)
        if overlap / (len(title_words) + 1) > 0.3:
            score += 5
            factors.append("✅ Headline matches content")
        else:
            score -= 5
            factors.append("⚠️ Headline may not match content")

    score = max(0, min(100, score))
    return score, factors

# News Analysis Section 
def extract_keywords(content):
    try:
        paragraphs = content.split("\n\n")
        initial_content = " ".join(paragraphs[:2])
        cleaned = re.sub(r"[^\w\s]", " ", initial_content)
        words = cleaned.split()
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "is", "are", "was", "were"
        }
        keywords = [word for word in words if word.lower() not in common_words]
        entities = [word for word in keywords if word[0].isupper()]
        search_terms = []
        if entities:
            search_terms.extend(entities[:3])
        if len(search_terms) < 3:
            remaining_words = [
                w for w in keywords if len(w) > 3 and w not in search_terms
            ]
            search_terms.extend(remaining_words[: 3 - len(search_terms)])
        return " ".join(search_terms)
    except Exception as e:
        return content.split(".")[0]

def calculate_relevance_score(article_title, query_terms):
    title_lower = article_title.lower()
    score = 0
    if any(
        query.lower() in title_lower
        for query in [" ".join(query_terms), "-".join(query_terms)]
    ):
        score += 1000
    for i, term in enumerate(query_terms):
        term_lower = term.lower()
        if term_lower in title_lower:
            score += 100
            score += 10 / (title_lower.index(term_lower) + 1)
    return score

def get_related_news(query):
    related_news = []
    query_terms = query.split()
    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
        news_api_results = newsapi.get_everything(
            q=query, language="en", sort_by="relevancy", page=1, page_size=10
        )
        if news_api_results["status"] == "ok":
            for article in news_api_results["articles"]:
                if any(term.lower() in article["title"].lower() for term in query.split()):
                    source_name = article.get("source", {}).get("name", "")
                    related_news.append({
                        "title": article["title"],
                        "url": article["url"],
                        "source": source_name,
                        "api_source": "NewsAPI",
                        "publishedAt": article.get("publishedAt", ""),
                        "description": article.get("description", ""),
                    })
    except Exception as e:
        pass
    try:
        google_news = GNews(language="en", max_results=10, country="US")
        gnews_results = google_news.get_news(query)
        if gnews_results:
            for article in gnews_results:
                try:
                    title = article["title"].get("default", "") if isinstance(article.get("title"), dict) else str(article.get("title", ""))
                    publisher = article["publisher"].get("default", "Unknown Source") if isinstance(article.get("publisher"), dict) else str(article.get("publisher", "Unknown Source"))
                    description = article["description"].get("default", "") if isinstance(article.get("description"), dict) else str(article.get("description", ""))
                    if title and article.get("url"):
                        related_news.append({
                            "title": title,
                            "url": str(article["url"]),
                            "source": publisher,
                            "api_source": "GNews",
                            "publishedAt": str(article.get("published date", "")),
                            "description": description,
                        })
                except Exception as e:
                    continue
    except Exception as e:
        pass
    seen_titles = set()
    filtered_news = []
    for article in related_news:
        try:
            title_key = str(article["title"]).lower()
            if not any(title_key in seen for seen in seen_titles):
                seen_titles.add(title_key)
                article["relevance_score"] = calculate_relevance_score(article["title"], query_terms)
                filtered_news.append(article)
        except Exception as e:
            continue
    filtered_news.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    return filtered_news

# Streamlit UI 
url = st.text_area("Enter URL")
if url:
    if not check_robots_txt(url):
        st.stop()
    try:
        with st.spinner("Analyzing article..."):
            content = c(url)
            if content == "Unable to extract content from the URL.":
                st.error("Could not extract content from the URL")
                st.stop()
                
            st.markdown("---")
            st.subheader("Scraped Article Content")
            with st.expander("Click to view article content", expanded=True):
                st.write(content)
               
            # Hybrid prediction
            model_confidence = hybrid_predict(content)
            
            # Credibility scoring
            article_title = content.split("\n")[0] if "\n" in content else ""
            source_score, credibility_factors = assess_source_credibility(url, content, title=article_title)
            
            # Combined score calculation
            combined_score = int(0.5 * model_confidence + 0.5 * source_score)

            st.markdown("---")
            st.subheader("Analysis Results")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(combined_score / 100)
            with col2:
                st.write(f"{combined_score}% Overall Confidence")
                with st.expander("Breakdown"):
                    st.write(f"**Model Prediction:** {model_confidence:.1f}%")
                    st.write(f"**Source Credibility:** {source_score}%")
                    st.markdown("### Credibility Factors")
                    for factor in credibility_factors:
                        st.write(f"- {factor}")

            # Related news section
            st.markdown("---")
            st.subheader("Related News Articles")
            query = extract_keywords(content)
            with st.spinner("Fetching related news..."):
                related_news = get_related_news(query)
            if related_news:
                for idx, article in enumerate(related_news):
                    try:
                        title_prefix = "🎯 " if article.get("relevance_score", 0) > 1000 else "📰 "
                        with st.expander(f"{title_prefix}{article['title']}", expanded=idx < 3):
                            st.markdown(f"""
                                **From:** {article['source']} (via {article['api_source']})  
                                **Published:** {article['publishedAt']}
                                {article['description'] if article.get('description') else 'No description available'}
                                [Read Full Article]({article['url']})
                                """)
                    except Exception as e:
                        continue
            else:
                st.warning("No related news articles found")
    except Exception as e:
        st.error(f"Error analyzing article: {str(e)}")
        st.write("Please make sure the URL is valid and accessible.")

st.markdown("---")
st.write("Steps to continue: ")
st.write("1. Find URL of News article.")
st.write("2. Copy and Paste the URL in text area.")
st.write("3. Press CTRL+Enter to run the model.")
st.write("4. Obtain results in slider and text form.")
st.write("5. Paste new URL over text area to re-use.")
st.write("6. Check other pages for more info and help.")

# Cleanup 
try:
    ArticleScraper().__del__()
except:
    pass
