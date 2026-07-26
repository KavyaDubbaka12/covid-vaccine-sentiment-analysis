import pandas as pd
import pickle
import re
import nltk

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords')

# Load dataset
data = pd.read_csv("vaccination_tweets.csv")
data = data[['text']]
data.columns = ['tweet']
data = data.dropna().drop_duplicates().head(5000)

# Better labeling
def get_sentiment(text):
    text = str(text).lower()

    if "not" in text:
        return "negative"
    elif any(word in text for word in ["bad", "worst", "pain", "side effect"]):
        return "negative"
    elif any(word in text for word in ["good", "great", "excellent", "safe"]):
        return "positive"
    else:
        return "neutral"

data["sentiment"] = data["tweet"].apply(get_sentiment)

# Clean text (NEGATION HANDLING)
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", str(text)).lower()
    words = text.split()

    new_words = []
    i = 0
    while i < len(words):
        if words[i] == "not" and i+1 < len(words):
            new_words.append("not_" + words[i+1])
            i += 2
        else:
            new_words.append(words[i])
            i += 1

    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")

    words = [w for w in new_words if w not in stop_words]

    return " ".join(words)

data["cleaned"] = data["tweet"].apply(clean_text)

# TF-IDF with BIGRAMS
vectorizer = TfidfVectorizer(ngram_range=(1,2))
X = vectorizer.fit_transform(data["cleaned"])
y = data["sentiment"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained successfully!")