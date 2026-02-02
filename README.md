# Persian Twitter Sentiment Analysis – Project Overview

This repository contains two main directories at the root level:

- **presentation/**  
  Materials used for the course presentation (slides, examples, etc.).  
  This folder is not part of the actual implementation pipeline.

- **project/**  
  The full implementation of the sentiment‑analysis workflow.  
  This is where all scripts are located and must be executed in order.

---

## 📁 Project Structure

Inside the **project/** directory, there are four subfolders.  
Each folder contains scripts that must be run **in sequence**:

---

### 1️⃣ `1.retrieveData/`

This folder contains the script responsible for **downloading the dataset** from:


moali-mkh-2000/PersianTwitterDataset-SentimentAnalysis


Running the script here will retrieve a collection of **Persian tweets** along with their sentiment labels.

---

### 2️⃣ `2.splitTrainTest/`

After retrieving the dataset, navigate to this folder.

Run the script here to **split the dataset into training and test sets**  
(typically an 80/20 split).

This step produces:

- `train.json`
- `test.json`

These files are used in the next stage.

---

### 3️⃣ `3.sentimentAnalysis/`

This folder contains **two separate training scripts**:

#### ✔ Baseline Model (SVM)
A simple TF‑IDF + SVM classifier.  
This serves as the baseline model for comparison.

#### ✔ BERT‑based Model
A transformer‑based model (fine‑tuned on the training set).  
This represents the more advanced approach.

Both scripts train their respective models and save them as `.pkl` files.

---

### 4️⃣ `4.comparisionWithRealWorld/`

This folder contains the **evaluation script**.

Running it will:

- load both trained models (SVM and BERT)
- run them on the `test.json` dataset
- print evaluation metrics for comparison

This step provides the final performance comparison between the baseline and the transformer‑based model.

---

## ✅ Summary of Execution Order

1. `project/1.retrieveData/` → download dataset  
2. `project/2.splitTrainTest/` → create train/test splits  
3. `project/3.sentimentAnalysis/` → train SVM and BERT models  
4. `project/4.comparisionWithRealWorld/` → evaluate and compare models  

---

Feel free to extend or modify the pipeline as needed for further experimentation.
