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
st.set_page_config(page_title="Sentiment Analyzer | محلل المشاعر", layout="centered")

# ========== Load Model and Vectorizer ========== #
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

# ========== Preprocessing Function ========== #
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# ========== App UI ========== #
st.title('🧠 Sentiment Analyzer | محلل المشاعر')

text_input = st.text_area("Enter your text (English/Arabic supported) | أدخل النص هنا", "I love this product!")

if st.button('Analyze | تحليل'):
    with st.spinner('Analyzing... | جاري التحليل...'):
        try:
            processed_text = preprocess_text(text_input)
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            # Result display
            if prediction == 1:
                st.success(f"✅ Positive | إيجابي (Confidence: {proba[1]*100:.1f}%)")
                st.balloons()
            else:
                st.error(f"❌ Negative | سلبي (Confidence: {proba[0]*100:.1f}%)")

            # Chart
            fig, ax = plt.subplots()
            ax.bar(['Negative', 'Positive'], proba, color=['red', 'green'])
            ax.set_ylim(0, 1)
            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ========== Footer ========== #
st.markdown("---")
st.caption("Developed by Data Analyst Mahmoud Tawfik")

# Additional info with links and description
st.markdown("""
---
### About the Developer

This sentiment analysis model was created by **Mahmoud Tawfik**, Data Analyst.

- 🔗 [LinkedIn Profile](https://www.linkedin.com/in/tawfeq)
- 🔗 [GitHub](https://github.com/Mahmoudmuhammedtawfik)

Thank you for using this app! """)
