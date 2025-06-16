import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt

# ========== Page Config (must be first command) ==========
st.set_page_config(page_title="Sentiment Analyzer | محلل المشاعر", layout="centered")

# ========== CSS for colorful design ==========
css = """
<style>
body {
    background-color: #eaf2f8;
}
h1, h2, h3, h4 {
    color: #105823;
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
.description-text {
    text-align: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #4b0082;
    font-weight: 600;
    margin-bottom: 2rem;
}
.footer {
    text-align: center;
    font-size: 0.9rem;
    color: #555;
    margin-top: 3rem;
}
.footer a {
    color: #4b0082;
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover {
    text-decoration: underline;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ========== NLTK stopwords check ==========
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ========== Load model and vectorizer ==========
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

# ========== Preprocessing ==========
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# ========== UI ==========
st.title('🧠 Sentiment Analyzer | محلل المشاعر')

st.markdown("""
<p class="description-text">
This app uses Natural Language Processing (NLP) to analyze text sentiment.<br>
التطبيق يستخدم معالجة اللغة الطبيعية لتحليل المشاعر.
</p>
""", unsafe_allow_html=True)

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

# ========== Footer ==========
st.markdown("---")
st.markdown("""
<div class="footer">
    Developed by Data Analyst <strong>Mahmoud Tawfik</strong><br>
    <a href="https://www.linkedin.com/in/tawfeq" target="_blank">LinkedIn</a> | 
    <a href="https://mahmoudmuhammedtawfik.github.io/portfolio/" target="_blank">Portfolio</a><br>
    © 2025 Sentiment Analysis App
</div>
""", unsafe_allow_html=True)
