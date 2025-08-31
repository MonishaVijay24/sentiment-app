# app.py
import streamlit as st
import joblib

# Load model and vectorizer
@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment:")

tweet_text = st.text_area("Tweet", "")

if st.button("Analyze"):
    if tweet_text.strip():
        tweet_vec = vectorizer.transform([tweet_text])
        prediction = model.predict(tweet_vec)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.success(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text.")
