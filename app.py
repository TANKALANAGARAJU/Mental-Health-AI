import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("mentalhealth_chatbot.pkl", "rb") as f:
    vectorizer, X, df = pickle.load(f)

# Function to get chatbot response
def get_chatbot_response(user_query):
    user_vec = vectorizer.transform([user_query])  # Convert input to vector
    similarities = cosine_similarity(user_vec, X)  # Compare with dataset questions
    best_match_idx = np.argmax(similarities)  # Find most similar question
    return df["answer"].iloc[best_match_idx]  # Return the best-matched answer

# Streamlit App
st.title("🧠 Mental Health AI Chatbot 🤖")
st.subheader("💬 Talk to the chatbot")

# User input
user_message = st.text_input("Enter your message:")

if st.button("Send"):
    if user_message.strip():
        response = get_chatbot_response(user_message)
        st.text_area("Bot:", value=response, height=350)
    else:
        st.warning("Please enter a message.")


