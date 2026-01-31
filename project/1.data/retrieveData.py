from datasets import load_dataset
import pandas as pd
import random

ds = load_dataset("moali-mkh-2000/PersianTwitterDataset-SentimentAnalysis")
data = ds["train"]

df = pd.DataFrame(data)

if "Unnamed: 0" in df.columns:
    df.rename(columns={"Unnamed: 0": "Id"}, inplace=True)

if "Numeric Labels" in df.columns:
    df.drop(columns=["Numeric Labels"], inplace=True)

feelings = ["Happy", "Sad", "Angry", "Neutral", "Intense Emotions"]
df["Person_Real_Mood"] = [random.choice(feelings) for _ in range(len(df))]

df.to_json("tweets_fa.json", orient="records", force_ascii=False, indent=2)