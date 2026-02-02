import json
import pandas as pd
import pickle
from hazm import Normalizer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

with open("../2.splitInTrainTest/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

normalizer = Normalizer()

def preprocess_text(text):
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    return " ".join(tokens)

df["processed_text"] = df["Tweets"].apply(preprocess_text)

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X = vectorizer.fit_transform(df["processed_text"])
y = df["Label"]

clf = LinearSVC()
clf.fit(X, y)

with open("svm_model.pkl", "wb") as f:
    pickle.dump((vectorizer, clf), f)
