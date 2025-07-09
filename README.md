# 📚 Kindle Review Sentiment Classifier

A machine learning web application that classifies book reviews from the Amazon Kindle Store as **Positive**, **Negative**, or **Neutral** using **Word2Vec embeddings**, **SMOTE** for class balancing, and **XGBoost** for classification.

---

## 🚀 Demo

👉 Try it live: [Streamlit App Link](https://kindle-review-sentiment-analysis-7qngdxfqylk6relirfbb8t.streamlit.app/)

---

## ✨ Project Highlights

- 🧠 Trained on real-world **Amazon Kindle book reviews** (Kaggle dataset)
- 💬 Converts raw reviews to numeric features using **Word2Vec**
- ⚖️ Handles class imbalance using **SMOTE**
- 🔍 Achieves **~83% accuracy** with **XGBoost**
- 🌐 Interactive UI built with **Streamlit**
- 📦 Deployed and shareable via public URL

---

## 🧪 Tech Stack

| Task                      | Tool / Library         |
|---------------------------|------------------------|
| Text Cleaning             | `nltk`, `BeautifulSoup`|
| Feature Engineering       | `Word2Vec (Gensim)`    |
| Balancing Dataset         | `imblearn.SMOTE`       |
| Classification Model      | `XGBoost`              |
| Model Deployment          | `Streamlit`            |
| Tokenization              | `nltk.word_tokenize`   |

---

## 📊 Model Performance

| Metric     | Value   |
|------------|---------|
| Accuracy   | **83.1%** |
| Precision  | 0.80–0.87 |
| Recall     | 0.78–0.88 |
| F1-Score   | 0.82–0.84 |

✅ Balanced performance on both classes (`Positive`, `Negative`, `Neutral`)

---

## 📁 Project Structure
kindle-review-sentiment-analysis/
├── app.py                    # Streamlit app interface
├── text_utils.py            # Preprocessing (cleaning + Word2Vec vectorizer)
├── xgboost_model_smote.pkl  # Trained XGBoost classifier with SMOTE
├── word2vec_model.model     # Trained Word2Vec model
├── requirements.txt         # Python dependencies for the app
└── README.md                # Project overview and usage instructions


---

## 🔧 Setup Instructions

### 1. Clone the repo

git clone https://github.com/vermavarsha/kindle-review-sentiment-analysis.git
cd kindle-review-sentiment-analysis

### 2.Install dependencies

pip install -r requirements.txt

### 3.Run locally

streamlit run app.py



