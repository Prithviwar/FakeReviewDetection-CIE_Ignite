from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import re

df = pd.read_csv('data/fake_reviews.csv')
df = df.rename(columns={'text_': 'review_text'})

def clean(t):
    return re.sub(r'[^a-z\s]', '', str(t).lower())

df['clean'] = df['review_text'].apply(clean)
X_tr, X_te, y_tr, y_te = train_test_split(df['clean'], df['label'], test_size=0.2, random_state=42)

tf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tr_t = tf.fit_transform(X_tr)
X_te_t = tf.transform(X_te)

m = LogisticRegression(max_iter=1000)
m.fit(X_tr_t, y_tr)

print('\nTrain Accuracy:', accuracy_score(y_tr, m.predict(X_tr_t)))
print('Test Accuracy:', accuracy_score(y_te, m.predict(X_te_t)))
