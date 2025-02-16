import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset (Ensure it has "question" and "answer" columns)
df = pd.read_csv("dataset\Mental_Health_FAQ.csv")

# Vectorize the questions using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Questions"])  # Convert questions to numerical form

# Save the trained model
with open("mentalhealth_chatbot.pkl", "wb") as f:
    pickle.dump((vectorizer, X, df), f)

print("✅ Model trained and saved as 'mentalhealth_chatbot.pkl'")
