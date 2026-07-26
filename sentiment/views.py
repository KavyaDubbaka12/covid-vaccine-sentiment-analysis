from django.shortcuts import render
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (only first time)
nltk.download('stopwords')

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# SAME CLEAN FUNCTION (as training)
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", str(text)).lower()
    words = text.split()

    new_words = []
    i = 0
    while i < len(words):
        if words[i] == "not" and i + 1 < len(words):
            new_words.append("not_" + words[i+1])
            i += 2
        else:
            new_words.append(words[i])
            i += 1

    stop_words = set(stopwords.words("english"))
    stop_words.discard("not")

    words = [w for w in new_words if w not in stop_words]

    return " ".join(words)

# MAIN VIEW
def home(request):
    result = None

    if request.method == "POST":
        text = request.POST.get("text", "").strip().lower()

        # Handle empty input
        if text == "":
            return render(request, "index.html", {
                "result": "Please enter text"
            })

        # RULE-BASED NEGATION FIX (IMPORTANT)
        if (
            "not good" in text or
            "not happy" in text or
            "not safe" in text or
            "not effective" in text or
            "not satisfied" in text or
            "sick" in text or
            "pain" in text
        ):
            result = "negative"

        else:
            cleaned = clean_text(text)
            vector = vectorizer.transform([cleaned])

            prediction = model.predict(vector)[0]
            result = prediction

    return render(request, "index.html", {
        "result": result
    })