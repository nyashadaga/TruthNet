import os
import requests
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlparse
import spacy
from collections import Counter

# Load environment variables
load_dotenv()

# NewsAPI Configuration
NEWS_API_KEYS = [
    os.getenv("NEWS_API_KEY_1"),
    os.getenv("NEWS_API_KEY_2"),
    os.getenv("NEWS_API_KEY_3"),
    os.getenv("NEWS_API_KEY_4"),
]

# Filter out any None values
NEWS_API_KEYS = [key for key in NEWS_API_KEYS if key]
if not NEWS_API_KEYS:
    st.error("Please set at least one NEWS_API_KEY in your environment variables")
    st.stop()

NEWS_API_URL = "https://newsapi.org/v2/everything"

# Track the current API key index
current_api_key_index = 0

# Add Indian news domains for filtering
INDIAN_NEWS_DOMAINS = [
    # Major English News
    "timesofindia.indiatimes.com",
    "thehindu.com",
    "hindustantimes.com",
    "indianexpress.com",
    "ndtv.com",
    "indiatoday.in",
    "news18.com",
    "moneycontrol.com",
    "livemint.com",
    "businessstandard.com",
    "economictimes.indiatimes.com",
    "financialexpress.com",
    "deccanherald.com",
    "tribuneindia.com",
    "telegraphindia.com",
    "thestatesman.com",
    "newindianexpress.com",
    "dnaindia.com",
    "outlookindia.com",
    "thequint.com",
    "scroll.in",
    "thewire.in",
    "firstpost.com",
    "theprint.in",
    "republicworld.com",
    # Hindi News
    "jagran.com",
    "bhaskar.com",
    "amarujala.com",
    "livehindustan.com",
    "navbharattimes.indiatimes.com",
    "zeenews.india.com",
    "aajtak.in",
    "abplive.com",
    "tv9hindi.com",
    "news24online.com",
    "punjabkesari.in",
    "patrika.com",
    "jansatta.com",
    "naidunia.com",
    "prabhatkhabar.com",
    "haribhoomi.com",
    # Regional Language News
    "mathrubhumi.com",  # Malayalam
    "manoramaonline.com",  # Malayalam
    "asianetnews.com",  # Malayalam
    "eenadu.net",  # Telugu
    "sakshi.com",  # Telugu
    "andhrajyothy.com",  # Telugu
    "dinamalar.com",  # Tamil
    "vikatan.com",  # Tamil
    "dailythanthi.com",  # Tamil
    "anandabazar.com",  # Bengali
    "sangbadpratidin.in",  # Bengali
    "bartamanpatrika.com",  # Bengali
    "lokmat.com",  # Marathi
    "maharashtratimes.com",  # Marathi
    "loksatta.com",  # Marathi
    "divyabhaskar.co.in",  # Gujarati
    "gujaratsamachar.com",  # Gujarati
    "sandesh.com",  # Gujarati
    "kannadaprabha.com",  # Kannada
    "prajavani.net",  # Kannada
    "udayavani.com",  # Kannada
    "punjabijagran.com",  # Punjabi
    "ajitjalandhar.com",  # Punjabi
    "rozanaspokesman.com",  # Punjabi
    "sanjevani.com",  # Kannada
    "samyuktakarnataka.com",  # Kannada
    "vijaykarnataka.com",  # Kannada
    "vishwavani.news",  # Telugu
    "ntnews.com",  # Telugu
    "dinamani.com",  # Tamil
    "dinakaran.com",  # Tamil
    "malayalam.samayam.com",  # Malayalam
    "deepika.com",  # Malayalam
    "deshabhimani.com",  # Malayalam
]

# Load spaCy model for semantic understanding
try:
    nlp = spacy.load("en_core_web_lg")  # Using the large model for better semantics
except:
    # Fallback to medium model if large isn't available
    nlp = spacy.load("en_core_web_md")


def get_next_api_key():
    """Get the next API key in rotation"""
    global current_api_key_index
    key = NEWS_API_KEYS[current_api_key_index]
    current_api_key_index = (current_api_key_index + 1) % len(NEWS_API_KEYS)
    return key


def calculate_semantic_similarity(headline1: str, headline2: str) -> float:
    """Calculate semantic similarity between two headlines"""
    try:
        # Process both headlines
        doc1 = nlp(headline1.lower())
        doc2 = nlp(headline2.lower())

        # Calculate similarity using spaCy's word vectors
        similarity = doc1.similarity(doc2)
        return similarity
    except:
        return 0.0


def get_related_news(headline: str) -> list:
    """Get related news articles with semantic matching"""
    all_articles = []
    seen_urls = set()

    try:
        api_key = get_next_api_key()

        # Extract key entities and concepts from headline
        doc = nlp(headline)
        key_entities = [ent.text for ent in doc.ents]
        key_nouns = [token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]]

        # Create broader search query
        search_terms = key_entities + key_nouns
        search_query = " OR ".join(search_terms) if search_terms else headline

        params = {
            "q": search_query,
            "apiKey": api_key,
            "language": "en",
            "pageSize": 100,  # Get more results for better matching
            "sortBy": "relevancy",
            "searchIn": "title,description",
        }

        response = requests.get(NEWS_API_URL, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])

            for article in articles:
                url = article.get("url")
                title = article.get("title", "")

                if url and url not in seen_urls and title:
                    seen_urls.add(url)

                    # Calculate semantic similarity
                    similarity = calculate_semantic_similarity(headline, title)

                    all_articles.append(
                        {
                            "title": title,
                            "url": url,
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "date": article.get("publishedAt", ""),
                            "description": article.get("description", ""),
                            "similarity": similarity,
                        }
                    )

    except Exception as e:
        st.error("Error fetching articles")

    # Sort by semantic similarity
    all_articles.sort(key=lambda x: x["similarity"], reverse=True)

    return all_articles


def display_articles(articles: list):
    """Display articles with similarity indication"""
    if not articles:
        st.warning("No related articles found")
        return

    st.write(f"Found {len(articles)} related articles")

    # Display highly similar articles first
    highly_similar = [a for a in articles if a["similarity"] > 0.7]
    moderately_similar = [a for a in articles if 0.4 <= a["similarity"] <= 0.7]
    somewhat_similar = [a for a in articles if a["similarity"] < 0.4]

    if highly_similar:
        st.subheader("Most Relevant Articles")
        for article in highly_similar:
            with st.expander(
                f"📰 {article['title']}", expanded=True
            ):  # Auto-expand highly similar
                display_article_content(article)

    if moderately_similar:
        st.subheader("Related Articles")
        for article in moderately_similar:
            with st.expander(f"📰 {article['title']}", expanded=False):
                display_article_content(article)

    if somewhat_similar:
        st.subheader("More Articles")
        for article in somewhat_similar:
            with st.expander(f"📰 {article['title']}", expanded=False):
                display_article_content(article)


def display_article_content(article):
    """Helper function to display individual article content"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.write(f"**Source:** {article['source']}")
        if article.get("description"):
            st.write(article["description"])
        st.write(f"[Read full article]({article['url']})")

    with col2:
        if article.get("date"):
            try:
                date = datetime.fromisoformat(article["date"].replace("Z", "+00:00"))
                st.write(f"📅 {date.strftime('%Y-%m-%d')}")
            except:
                pass


def main():
    st.title("📰 Related News Finder")
    st.write("Enter a news headline to find semantically related articles.")

    headline = st.text_input("Enter a news headline:")

    if headline:
        with st.spinner("Searching for related articles..."):
            articles = get_related_news(headline)
            display_articles(articles)


if __name__ == "__main__":
    main()
