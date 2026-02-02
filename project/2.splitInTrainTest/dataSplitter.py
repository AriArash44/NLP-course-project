import json
import pandas as pd
from sklearn.model_selection import train_test_split

with open("../1.data/tweets_fa.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df.to_json("train.json", orient="records", force_ascii=False, indent=2)
test_df.to_json("test.json", orient="records", force_ascii=False, indent=2)
