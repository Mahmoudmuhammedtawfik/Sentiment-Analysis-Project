
import streamlit as st
st.set_page_config(page_title="محلل المشاعر", layout="centered")

import pickle
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer

# تحميل stopwords
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# تحميل النموذج والفيكتورايزر
@st.cache_resource
def load_components():
    model = pickle.load(open('trained_model.sav', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    return model, vectorizer

model, vectorizer = load_components()

# دالة تنظيف النصوص
def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# واجهة المستخدم
st.title('🧠 محلل المشاعر الآلي')
text_input = st.text_area("أدخل النص هنا:", "I love this product!")

if st.button('تحليل'):
    with st.spinner('جاري التحليل...'):
        try:
            processed_text = preprocess_text(text_input)
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            if prediction == 1:
                st.success(f"✅ إيجابي (ثقة: {proba[1]*100:.1f}%)")
                st.balloons()
            else:
                st.error(f"❌ سلبي (ثقة: {proba[0]*100:.1f}%)")

            fig, ax = plt.subplots()
            ax.bar(['سلبي', 'إيجابي'], proba, color=['red', 'green'])
            ax.set_ylim(0, 1)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")

st.markdown("---")
st.caption("تطبيق تحليل المشاعر باستخدام التعلم الآلي")
