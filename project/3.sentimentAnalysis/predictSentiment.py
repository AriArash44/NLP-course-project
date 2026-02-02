import json
import pandas as pd
import pickle
from hazm import Normalizer, word_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder

with open("../2.splitInTrainTest/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

normalizer = Normalizer()

def preprocess_text(text):
    text = normalizer.normalize(text)
    tokens = word_tokenize(text)
    return " ".join(tokens)

df["processed_text"] = df["Tweets"].apply(preprocess_text)

le = LabelEncoder()
df["label_id"] = le.fit_transform(df["Label"])

model_name = "HooshvareLab/bert-fa-base-uncased-sentiment-snappfood"

tokenizer = AutoTokenizer.from_pretrained(model_name)

class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = TweetDataset(df["processed_text"].tolist(), df["label_id"].tolist())

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(le.classes_),
    ignore_mismatched_sizes=True
)

training_args = TrainingArguments(
    output_dir="./bert_output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=50,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

with open("bert_model.pkl", "wb") as f:
    pickle.dump((model, tokenizer, le), f)
