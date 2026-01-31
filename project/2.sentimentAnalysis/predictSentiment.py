import json
import pandas as pd
from hazm import Normalizer, word_tokenize
from transformers import pipeline

with open("../1.data/tweets_fa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

normalizer = Normalizer()

def preprocess_text(text):
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    return " ".join(tokens)

df["processed_text"] = df["Tweets"].apply(preprocess_text)

classifier = pipeline(
    "sentiment-analysis",
    model="HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"
)

def map_to_emotion(label, score):
    label = label.lower()
    
    if "positive" in label:
        return "Happy"
    elif "negative" in label:
        if score > 0.85:
            return "Intense Emotions"
        return "Sad"
    else:
        return "Neutral"

predicted = []

for text in df["processed_text"]:
    result = classifier(text[:512])[0]
    emotion = map_to_emotion(result["label"], result["score"])
    predicted.append(emotion)

df["Predicted_Feeling"] = predicted

output_df = df[["Id", "Tweets", "Label", "Predicted_Feeling"]]

output_df.to_json(
    "tweets_with_emotions.json",
    orient="records",
    force_ascii=False,
    indent=2
)
