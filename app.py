import streamlit as st
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Spam Detector Pro",
    page_icon="📩",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

body {
    background-color: #0E1117;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    background: #cc261b;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.stTextArea textarea {
    background-color: #1c1f26;
    color: white;
    border-radius: 12px;
    border: 1px solid #333;
}

.stButton>button {
    width: 100%;
    height: 50px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
    background: linear-gradient(90deg, #ff4b4b, #ff0000);
    color: white;
    border: none;
}

.stButton>button:hover {
    background: linear-gradient(90deg, #ff0000, #ff4b4b);
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<div class='main-title'>📩 Spam Detector Pro</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Powered by TF-IDF & Multinomial Naive Bayes</div>", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 About This App")
st.sidebar.info("""
This application detects whether an Email or SMS message is Spam or Not Spam.

🔹 NLP Preprocessing  
🔹 TF-IDF Vectorization  
🔹 Multinomial Naive Bayes  
🔹 Built with Streamlit  

Created by Shahrier Sabit 🚀
""")

# ------------------ NLP SETUP ------------------
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    stop_words = set(stopwords.words('english'))

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ------------------ LOAD MODEL ------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ------------------ INPUT AREA ------------------
input_sms = st.text_area("✉ Enter your message here", height=150)

# ------------------ BUTTON ------------------
if st.button("🔍 Classify Message"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")
    else:
        transformed_text = transform_text(input_sms)
        vectorized_text = tfidf.transform([transformed_text])
        result = model.predict(vectorized_text)

        if result[0] == 1:
            st.markdown(
                "<div class='result-box' style='background-color:#3a0d0d; color:#ff4b4b;'>🚨 This message is SPAM</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result-box' style='background-color:#0d3a1a; color:#00ff88;'>✅ This message is NOT SPAM</div>",
                unsafe_allow_html=True
            )

# ------------------ FOOTER ------------------
st.markdown("<div class='footer'>© 2026 Shahrier Sabit | Machine Learning Project</div>", unsafe_allow_html=True)