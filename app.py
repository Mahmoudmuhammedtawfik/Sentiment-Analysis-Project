import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt

# ========== Page Config (Must be FIRST Streamlit command) ==========
st.set_page_config(page_title="Sentiment Analyzer | محلل المشاعر", layout="centered")

# ========== CSS for colorful design + math background ==========
css = """
<style>
.math-background {
    background-color: #f5f5f5; /* light background */
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
    font-family: "Times New Roman", Times, serif;
    max-width: 400px;
    margin: 1rem auto 2rem auto;
    position: relative;
}
.math-background::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: linear-gradient(135deg, rgba(74, 20, 140, 0.05) 0%, rgba(0, 0, 0, 0) 50%);
    z-index: -1;
    border-radius: 8px;
}
body {
    background-color: #eaf2f8;
}
h1, h2, h3, h4 {
    color: #4b0082;
    font-weight: 700;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.stButton>button {
    background: linear-gradient(45deg, #7b2ff7, #f107a3);
    color: white;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.5rem 1.5rem;
    transition: background 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(45deg, #f107a3, #7b2ff7);
}
.result-positive {
    color: green;
    font-weight: 700;
    font-size: 1.2rem;
}
.result-negative {
    color: red;
    font-weight: 700;
    font-size: 1.2rem;
}
</style>
"""

# Inject CSS styles
st.markdown(css, unsafe_allow_html=True)

# NLTK stopwords check & download if missing
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Load model and vectorizer (cached)
@st.cache_resource
def load_components():
    try:
        model = pickle.load(open('trained_model.sav', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading components: {str(e)}")
        st.stop()

model, vectorizer = load_components()

# Preprocess input text
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# UI title
st.title('🧠 Sentiment Analyzer | محلل المشاعر')

# Simple description inside the math-background div (no icons)
st.markdown("""
<div class="math-background">
    <p>This app uses Natural Language Processing (NLP) to analyze text sentiment.<br>
    التطبيق يستخدم معالجة اللغة الطبيعية لتحليل المشاعر.</p>
</div>
""", unsafe_allow_html=True)

# Text input area
text_input = st.text_area("Enter your text (English/Arabic supported) | أدخل النص هنا:", "I love this product!")

if st.button('Analyze | تحليل'):
    with st.spinner('Analyzing... | جاري التحليل...'):
        try:
            processed_text = preprocess_text(text_input)
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            if prediction == 1:
                st.markdown(f'<p class="result-positive">✅ Positive | إيجابي (Confidence: {proba[1]*100:.1f}%)</p>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<p class="result-negative">❌ Negative | سلبي (Confidence: {proba[0]*100:.1f}%)</p>', unsafe_allow_html=True)

            fig, ax = plt.subplots()
            ax.bar(['Negative', 'Positive'], proba, color=['red', 'green'])
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer with your info and links
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:0.9rem; color:#555;">
    Developed by Data Analyst <strong>Mahmoud Tawfik</strong><br>
    <a href="https://www.linkedin.com/in/tawfeq" target="_blank">LinkedIn</a> | 
    <a href="https://mahmoudmuhammedtawfik.github.io/portfolio/" target="_blank">Portfolio</a><br>
    © 2025 Sentiment Analysis App
</div>
""", unsafe_allow_html=True)
