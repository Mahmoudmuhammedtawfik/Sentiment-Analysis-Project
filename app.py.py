import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

# حل مشكلة stopwords - الطريقة الأفضل
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

# تحميل النموذج مع معالجة الأخطاء
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('trained_model.sav', 'rb'))
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        return model, vectorizer
    except Exception as e:
        st.error(f"حدث خطأ في تحميل النموذج: {str(e)}")
        st.stop()

model, vectorizer = load_model()

# دالة معالجة النص المحسنة
def preprocess_text(text):
    ps = PorterStemmer()
    # إزالة الروابط، علامات @، والأحرف الخاصة
    text = re.sub(r'http\S+|www\S+|@\w+|[^\w\s]', '', text)
    text = text.lower().strip()
    words = text.split()
    # استخدام stopwords من NLTK أو sklearn كبديل
    words = [ps.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# واجهة Streamlit
st.set_page_config(page_title="محلل المشاعر", layout="wide")

# عنوان التطبيق
st.title('📊 محلل المشاعر الآلي')
st.markdown("""
<style>
.positive { color: green; font-weight: bold; }
.negative { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# إدخال النص
user_input = st.text_area("أدخل النص الذي تريد تحليله:", 
                         "I love this product! It's amazing!")

if st.button('تحليل المشاعر', type="primary"):
    with st.spinner('جاري التحليل...'):
        try:
            # المعالجة والتنبؤ
            processed_text = preprocess_text(user_input)
            X = vectorizer.transform([processed_text])
            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            confidence = np.max(proba) * 100
            
            # عرض النتائج
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("النتيجة:")
                if prediction == 1:
                    st.markdown(f'<p class="positive">✅ إيجابي (ثقة: {confidence:.2f}%)</p>', 
                               unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.markdown(f'<p class="negative">❌ سلبي (ثقة: {confidence:.2f}%)</p>', 
                               unsafe_allow_html=True)
            
            with col2:
                # رسم بياني لاحتمالات التنبؤ
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.bar(['سلبي', 'إيجابي'], proba, color=['red', 'green'])
                ax.set_ylim(0, 1)
                ax.set_title('احتمالات التنبؤ')
                st.pyplot(fig)
            
            # تفاصيل إضافية
            with st.expander("عرض التفاصيل الفنية"):
                st.write("**النص بعد المعالجة:**", processed_text)
                st.write("**احتمالية الإيجابي:**", f"{proba[1]:.2%}")
                st.write("**احتمالية السلبي:**", f"{proba[0]:.2%}")
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء التحليل: {str(e)}")

# تذييل الصفحة
st.markdown("---")
st.caption("تطبيق تحليل المشاعر باستخدام خوارزميات التعلم الآلي - تم التطوير بواسطة [اسمك]")