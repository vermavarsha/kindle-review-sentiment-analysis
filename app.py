import streamlit as st
import joblib
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from text_utils import avg_word2vec,clean_text
import numpy as np
import nltk

@st.cache_resource
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')

download_nltk_resources()


# Setup
st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("xgboost_model_smote.pkl")
    w2v_model = Word2Vec.load("word2vec_model.model")
    return model, w2v_model

model, w2v_model = load_models()

# UI
st.title("📚 Kindle Review Sentiment Classifier")
st.write("Enter a book review below to see if it's Positive or Negative.")

user_input = st.text_area("✍️ Your Review", height=150)

if st.button("Predict"):
    if user_input.strip():
        cleaned = clean_text(user_input)
        tokens = word_tokenize(cleaned)
        vector = avg_word2vec(tokens, w2v_model).reshape(1, -1)
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0][1]  # Prob of class 1 (Positive)
        # sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"
        if probability > 0.75:
            sentiment = "🟢 Positive"
        elif probability < 0.25:
            sentiment = "🔴 Negative"
        else:
            sentiment = "🔵 Neutral"
        st.subheader(f"Prediction: {sentiment}")
        st.info(f"🔍 Confidence: **{probability * 100:.2f}%**")
    else:
        st.warning("Please enter a review to classify.")
