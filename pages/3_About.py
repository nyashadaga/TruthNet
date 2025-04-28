import streamlit as st

# Set basic page config
st.set_page_config(page_title="About TruthNet", layout="wide")

def main():
    st.title("About TruthNet")

    # Brief Description of the Solution
    st.header("Brief Description of the Solution")
    st.markdown("""
**Objective:**  
Detect and analyze news articles' authenticity as a percentage score using machine learning and NLP, with cross-checking via API integration and web scraping.

**Key Components:**
- **Machine Learning Models Used for Classification:**  
  - Multinomial Naive Bayes (NB)  
  - K-Nearest Neighbors (KNN)  
  - Logistic Regression (LR)  
  - Passive Aggressive Classifier (PA)

- **Feature Extraction Technique:**  
  - Term Frequency–Inverse Document Frequency (TF-IDF)

- **Data Collection Process:**  
  - Automated web scraping extracts news articles from various sources.

- **API Integration for Cross-Checking:**  
  - API-based validation is used to verify article authenticity.

- **Model Evaluation & Comparative Study:**  
  - Multiple classifiers are compared to determine the best-performing model.
  - Instead of classifying news as simply real or fake, **TruthNet provides a percentage score of authenticity**.
    """)

    # How it works
    st.header("How it Works")
    st.markdown("""
1. Enter a news article URL.
2. TruthNet scrapes the article content.
3. The system extracts features and analyzes the content using trained machine learning models.
4. API cross-checking is performed for additional validation.
5. The user receives an authenticity percentage and can view related news articles for comparison.
    """)

    # API Credits
    st.header("API Credits")
    st.markdown("""
TruthNet uses the following news APIs for cross-verification and related news:
- NewsAPI
- GNews

We thank these services for providing access to their news data.
    """)

    # Technical Information
    st.header("Technical Information")
    st.markdown("""
- Built with Python and Streamlit
- Uses scikit-learn for machine learning models
- Uses TensorFlow/Keras for deep learning (BiLSTM)
- Uses NLTK and spaCy for natural language processing
- Real-time web scraping and news verification
- TF-IDF for feature extraction
- Hybrid model for improved accuracy and confidence
    """)

if __name__ == "__main__":
    main()
