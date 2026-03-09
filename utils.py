import re
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Session review history
# -----------------------------
# Dictionary storing reviews per client session
# e.g. {"default": ["review 1", "review 2"]}
session_reviews = {}

#similarity vectorizer
similarity_vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2
)

# -----------------------------
# Load reference reviews
# -----------------------------
reference_df = pd.read_csv("data/fake_reviews.csv")

# Rename to match
reference_df = reference_df.rename(columns={"text_": "review_text"})

# Take a small random sample for similarity checking
REFERENCE_SAMPLE_SIZE = 500
reference_texts = random.sample(
    list(reference_df['review_text']), 
    REFERENCE_SAMPLE_SIZE
)

# Fit on reference reviews
similarity_vectorizer.fit(reference_texts)

#------------------------------
# nltk
#------------------------------

nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

stop_words = set(stopwords.words('english'))
sia = SentimentIntensityAnalyzer()

# -----------------------------
# Load model & vectorizer
# -----------------------------
model = joblib.load("model/fake_review_model.pkl")
tfidf = joblib.load("model/tfidf_vectorizer.pkl")

# -----------------------------
# Text Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def ml_prediction(review):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0][1]  # fake probability
    return prediction, probability

def sentiment_penalty(review):
    sentiment = sia.polarity_scores(review)['compound']
    if sentiment > 0.85 or sentiment < -0.85:
        return 10, "Extremely emotional language"
    return 0, None

def promotion_penalty(review):
    tokens = nltk.word_tokenize(review)
    pos_tags = nltk.pos_tag(tokens)

    # Superlatives (JJS = Adjective, superlative; RBS = Adverb, superlative)
    superlatives = [word for word, pos in pos_tags if pos in ['JJS', 'RBS']]
    
    # Exclamation mark abuse
    exclamation_count = review.count('!')

    # If it's heavily superlative or combines superlatives with excessive exclamations
    if len(superlatives) >= 3:
        return 20, "Overly promotional wording (excessive superlatives)"
    if len(superlatives) >= 2 and exclamation_count >= 3:
        return 15, "Promotional wording with high exclamations"
        
    return 0, None

def low_information_penalty(review):
    word_count = len(review.split())

    if word_count < 4:
        return 20, "Extremely short review with no details"
    elif word_count < 8:
        return 15, "Low-information generic review"
    
    return 0, None


def duplicate_penalty(review, client_id="default", explicit_history=None):
    # Use explicit history if provided (for batches), otherwise use session history
    if explicit_history is not None:
        history = explicit_history
    else:
        if client_id not in session_reviews:
            session_reviews[client_id] = []
        history = session_reviews[client_id]

    if not history:
        if explicit_history is None:
            session_reviews[client_id].append(review)
        return 0, None

    input_vec = similarity_vectorizer.transform([review])
    history_vecs = similarity_vectorizer.transform(history)

    similarities = cosine_similarity(input_vec, history_vecs)[0]
    max_similarity = max(similarities)

    penalty = 0
    reason = None
    if max_similarity > 0.8:
        # Heavily penalize exact duplicates within the same session/batch
        penalty = 60 
        reason = "Duplicate or near-duplicate review detected within session"
    elif max_similarity > 0.65:
        penalty = 30
        reason = "Review is similar to previous submissions in session"

    # Only append to session history if we aren't using explicit batch history
    if explicit_history is None:
        session_reviews[client_id].append(review)
        
    return penalty, reason


def rating_sentiment_penalty(review, rating):
    sentiment = sia.polarity_scores(review)['compound']

    # Very positive text but low rating
    if rating <= 2 and sentiment > 0.7:
        return 40, "Positive text but low rating"

    # Very negative text but high rating
    if rating >= 4 and sentiment < -0.7:
        return 40, "Negative text but high rating"

    # Extremely positive sentiment with 5-star rating
    if rating == 5 and sentiment > 0.9:
        return 15, "Excessive positivity for maximum rating"

    return 0, None

def generic_phrase_penalty(review):
    tokens = nltk.word_tokenize(review)
    pos_tags = nltk.pos_tag(tokens)

    # Count Nouns vs Adjectives/Adverbs
    nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
    adj_adv = [word for word, pos in pos_tags if pos in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']]

    # Filter out meaning-less nouns often used by bots
    meaningful_nouns = [n.lower() for n in nouns if n.lower() not in ['product', 'item', 'thing', 'stuff']]

    # Zero descriptive nouns
    if len(meaningful_nouns) == 0:
        if len(adj_adv) >= 2:
            return 20, "Highly generic structure (vague adjectives, no specific nouns)"
        return 15, "Generic structure lacking specific nouns"

    # Extremely high ratio of fluff to actual substance
    if len(adj_adv) > 0 and len(meaningful_nouns) > 0:
        ratio = len(adj_adv) / len(meaningful_nouns)
        if ratio >= 4.0:
            return 20, "Highly generic phrasing (overloaded descriptors)"

    return 0, None

def score_review(review, rating, client_id="default", explicit_history=None):
    print("DEBUG: Rating received:", rating)
    score = 100
    reasons = []

    # ML probability
    _, ml_prob = ml_prediction(review)
    if ml_prob > 0.75:
        score -= 35
        reasons.append("High ML fake probability")

    # Rule-based checks (Independent)
    for penalty_func in [low_information_penalty, generic_phrase_penalty]:
        penalty, reason = penalty_func(review)
        if penalty > 0 and reason is not None:
            score -= penalty
            reasons.append(reason)
            
    # Grouped Checks (Sentiment & Promotion)
    # We evaluate these together to prevent "penalty stacking" on highly emotional/promotional reviews
    sent_pen, sent_reason = sentiment_penalty(review)
    promo_pen, promo_reason = promotion_penalty(review)
    rate_pen, rate_reason = rating_sentiment_penalty(review, rating)
    
    # Isolate mismatches (both positive text + low rating, and negative text + high rating)
    mismatch_penalty = 0
    mismatch_reason = None
    if rate_pen > 0 and rate_reason in ["Negative text but high rating", "Positive text but low rating"]:
        mismatch_penalty = rate_pen
        mismatch_reason = rate_reason
    
    # Isolate positive/hype penalties
    positive_flags = []
    if sent_pen > 0: positive_flags.append((sent_pen, sent_reason))
    if promo_pen > 0: positive_flags.append((promo_pen, promo_reason))
    if rate_pen > 0 and rate_reason == "Excessive positivity for maximum rating":
        positive_flags.append((rate_pen, rate_reason))

    # Apply only the WORST positive/hype penalty to the score, but add ALL reasons for transparency
    if positive_flags:
        worst_penalty = max(flag[0] for flag in positive_flags)
        score -= int(worst_penalty)
        for flag in positive_flags:
            reasons.append(flag[1])
        
    # Apply the mismatch penalty independently if it exists
    if mismatch_penalty > 0 and mismatch_reason is not None:
        score -= int(mismatch_penalty)
        reasons.append(mismatch_reason)

    # Duplicate Check (Needs special handling)
    dup_penalty, dup_reason = duplicate_penalty(review, client_id, explicit_history)
    if dup_penalty > 0 and dup_reason is not None:
        score -= int(dup_penalty)
        reasons.append(dup_reason)

    # Assign flag
    if score >= 75:
        flag = "GREEN"
    elif score >= 50:
        flag = "YELLOW"
    else:
        flag = "RED"

    return {
        "score": score,
        "flag": flag,
        "ml_fake_probability": round(ml_prob, 2),
        "reasons": reasons
    }

#temporary test case
if __name__ == "__main__":
    review = "This pillow is very comfortable and provides excellent support for my neck."
    rating = 5
    result = score_review(review, rating, client_id="test_user")
    print(result)
    review = "This pillow is very comfortable and provides excellent support for my neck!"
    result = score_review(review, rating, client_id="test_user")
    print(result)
