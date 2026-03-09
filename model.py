import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# -----------------------------
# Setup
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("data/fake_reviews.csv")

# Rename for clarity
df = df.rename(columns={"text_": "review_text"})

print(df.head())
print(df.columns)
print(df.shape)

# -----------------------------
# Cleaning Function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['cleaned_review'] = df['review_text'].apply(clean_text)

# -----------------------------
# Train-Test Split
# -----------------------------
X = df['cleaned_review']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# -----------------------------
# Model Training
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save Model & Vectorizer
# -----------------------------
joblib.dump(model, "model/fake_review_model.pkl")
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")

probs = model.predict_proba(X_test_tfidf)[:, 1]

print("Min:", np.min(probs))
print("Max:", np.max(probs))
print("Mean:", np.mean(probs))
print("90th percentile:", np.percentile(probs, 90))
print("95th percentile:", np.percentile(probs, 95))
