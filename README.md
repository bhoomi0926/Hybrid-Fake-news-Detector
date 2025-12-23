# 📰 Hybrid Fake News Detection System  
### Machine Learning (PAC + TF-IDF) + Confidence Threshold + Live Verification (NewsAPI)

This project implements a **hybrid fake news classification system** that first uses a machine-learning model to classify news headlines and, whenever the model is not confident, it falls back to **live news verification using NewsAPI**.  
This gives **higher accuracy**, **real-world reliability**, and improves trustworthiness of results.

---

## 📂 Project Structure

Final+NewsAPI/
│
├── Dataset/
│ ├── Fake.csv
│ └── True.csv
│
├── model/
│ ├── fake_news_model.joblib
│ └── vectorizer.joblib
│
├── venv/ # Virtual environment (auto-created)
│
├── app.py # Main hybrid detection application
├── train_model.py # ML model training script
├── requirements.txt # Dependencies
└── README.md # Documentation (this file)

---

## ⭐ Project Overview

This system combines:

### **1. Machine Learning Prediction**
- Uses TF-IDF vectorization  
- Trained with **Passive Aggressive Classifier (PAC)**  
- Dataset: `True.csv` + `Fake.csv`  
- Model outputs:
  - **Real**
  - **Fake**
  - **Uncertain (low confidence)**  

---

### **2. Confidence Threshold Logic**
The classifier generates a confidence score using the decision function.

- If **confidence ≥ threshold → Use ML prediction**
- If **confidence < threshold → Trigger fallback**

This prevents incorrect predictions and improves safety.

---

### **3. NewsAPI Fallback System**
If ML cannot confidently classify, the system queries **NewsAPI** to search for live published news.

- Fetches latest articles
- Displays title + description
- Helps verify real-world credibility

---

## 🚀 How to Run the Project

### **1. Install Dependencies**

pip install -r requirements.txt
(Dependency list from: 
requirements
)

### **2. Add Your NewsAPI Key
Create a .env file:

ini
NEWS_API_KEY=your_key_here
You can generate a free key at:
https://newsapi.org/

### **3. Train the Machine Learning Model
(Only needed if you want to retrain)

python train_model.py
This will generate:
model/fake_news_model.joblib
model/vectorizer.joblib

Training workflow implemented in:
train_model

### **4. Run the Hybrid Detector Application

python app.py
Gradio UI will open in your browser with:

Text input

ML output

Confidence score

NewsAPI fallback output if needed

Implemented in:

app

### 🧠 How the System Works (Step-by-Step)
User enters a news headline

TF-IDF converts the text into numeric vectors

PAC classifier makes a prediction

If confidence is high → returns Real/Fake

If confidence is low →

Searches NewsAPI

Displays first relevant headline

Returns UNCERTAIN → Using NewsAPI

### 📊 Dataset Used
Located in Dataset/:

Fake.csv — Fake news headlines
True.csv — Real verified headlines

The two datasets are merged, labeled, shuffled, and used to train the model.

(Referenced in training script: 
train_model
)

### 🏗 Technologies Used
Component-> Technology
Model-> Passive Aggressive Classifier
Vectorizer-> TF-IDF
Web UI-> Gradio
Live Verification-> NewsAPI
Storage-> Joblib Model + Vectorizer

### 📌 Features
✔ Hybrid ML + API approach
✔ Confidence-based decision routing
✔ Real-time headline verification
✔ Clean Gradio interface
✔ Reliable and scalable architecture

### 📈 Future Improvements
Fine-tuning with Transformers (BERT, RoBERTa)

Multilingual support

Full news article analysis

Deploy via HuggingFace or Docker

Chrome extension version

Real-time misinformation dashboard

### 🙌 Acknowledgements
Dataset sources: Fake & True CSV files

NewsAPI for real-time news verification

Gradio for interactive UI
