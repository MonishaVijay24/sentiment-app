import streamlit as st
import joblib

# Load the model and vectorizer
try:
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run the training notebook first.")
    st.stop()

st.title("Tweet Sentiment Analysis")
st.write("Enter a tweet below to analyze its sentiment.")

tweet_text = st.text_area("Tweet", "")

if st.button("Analyze"):
    if tweet_text:
        tweet_vec = vectorizer.transform([tweet_text])
        prediction = model.predict(tweet_vec)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text to analyze.")

