
import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt

# ========== NLTK stopwords check ========== #
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ========== Page Config (must be first Streamlit command) ========== #
st.set_page_config(page_title="🧠 Sentiment Analyzer | محلل المشاعر", layout="centered")

# ========== Custom CSS for colorful design and styling ========== #
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #fceabb 0%, #f8b500 100%);
        color: #333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea>div>textarea {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        font-size: 16px;
        padding: 10px;
    }
    footer, header, .css-18e3th9 {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ========== Load Model and Vectorizer with caching ========== #
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

# ========== Text Preprocessing Function ========== #
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# ========== App UI ========== #
st.title('🧠 Sentiment Analyzer | محلل المشاعر')
st.markdown("## 📝 Enter text to analyze its sentiment (English/Arabic supported)")

text_input = st.text_area("💬 Your text here... | أدخل النص هنا", "I love this product!")

if st.button('🔍 Analyze | تحليل'):
    with st.spinner('🔄 Analyzing... | جاري التحليل...'):
        try:
            processed_text = preprocess_text(text_input)
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            if prediction == 1:
                st.success(f"✅ Positive | إيجابي (Confidence: {proba[1]*100:.1f}%) 🎉")
                st.balloons()
            else:
                st.error(f"❌ Negative | سلبي (Confidence: {proba[0]*100:.1f}%) 😞")

            fig, ax = plt.subplots()
            ax.bar(['Negative', 'Positive'], proba, color=['#FF4B4B', '#4CAF50'])
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ========== Footer ========== #
st.markdown("---")
st.caption("This sentiment analysis model was created by **Data Analyst Mahmoud Tawfik** into insights for impactful decisions Transforming data 💡")

# Additional info with links and description
st.markdown("""

- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/tawfeq)
- 🔗 [GitHub](https://github.com/Mahmoudmuhammedtawfik)

Thank you for using this app! """)
