import json
import pandas as pd
import pickle
from hazm import Normalizer, word_tokenize
from sklearn.metrics import classification_report
import torch

with open("../2.splitInTrainTest/test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

normalizer = Normalizer()

def preprocess_text(text):
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    return " ".join(tokens)

df["processed_text"] = df["Tweets"].apply(preprocess_text)

with open("../3.sentimentAnalysis/svm_model.pkl", "rb") as f:
    vectorizer, svm_model = pickle.load(f)

X_test_svm = vectorizer.transform(df["processed_text"])
svm_pred = svm_model.predict(X_test_svm)

print("=== SVM Evaluation ===")
print(classification_report(df["Label"], svm_pred))

with open("../3.sentimentAnalysis/bert_model.pkl", "rb") as f:
    bert_model, tokenizer, label_encoder = pickle.load(f)

texts = df["processed_text"].tolist()
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

with torch.no_grad():
    outputs = bert_model(**encodings)
    logits = outputs.logits
    bert_pred_ids = torch.argmax(logits, dim=1).numpy()

bert_pred = label_encoder.inverse_transform(bert_pred_ids)

print("=== BERT Evaluation ===")
print(classification_report(df["Label"], bert_pred))
