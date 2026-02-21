# 📩 Spam Detector Pro

A Machine Learning powered SMS & Email Spam Detection web application built using Natural Language Processing (NLP) and deployed with Streamlit.

This project classifies messages as **Spam 🚨** or **Not Spam ✅** using TF-IDF vectorization and Multinomial Naive Bayes.

---

## 🚀 Live Demo https://spam-detector-app-sbt.streamlit.app/

---

# 📌 Project Overview

Spam messages are one of the most common problems in digital communication.
This project builds a complete end-to-end machine learning pipeline to detect spam messages accurately.

### Workflow:

1. Text preprocessing using NLP  
2. Feature extraction using TF-IDF  
3. Model comparison (GaussianNB, MultinomialNB, BernoulliNB)  
4. Selecting the best performing model  
5. Deploying the trained model using Streamlit  

---

# 📊 Dataset

- File: `spam.csv`
- Type: SMS Spam Dataset  
- Target column:
  - `0` → Not Spam (Ham)  
  - `1` → Spam  

---

# 🧠 Machine Learning Pipeline

## 1️⃣ Text Preprocessing

Performed using **NLTK**:

- Convert text to lowercase  
- Tokenization  
- Remove special characters  
- Remove stopwords  
- Stemming using PorterStemmer  

Function used:

```python
transform_text(text)
```

---

## 2️⃣ Feature Engineering

Used:

- **TF-IDF Vectorizer**
- `max_features = 3000`

This converts processed text into numerical vectors for model training.

---

## 3️⃣ Model Training & Comparison

Models tested:

- Gaussian Naive Bayes  
- Multinomial Naive Bayes  
- Bernoulli Naive Bayes  

### ✅ Final Selected Model:
**Multinomial Naive Bayes**

Reason:

- Higher precision  
- Better suited for text classification  
- Performs well with TF-IDF features  

---

# 📈 Model Evaluation

Metrics used:

- Accuracy  
- Precision  
- Confusion Matrix  

MultinomialNB showed the best overall performance and was selected for deployment.

---

# 💾 Model Saving

The trained model and vectorizer were saved using pickle:

```python
vectorizer.pkl
model.pkl
```

---

# 🌐 Streamlit Web Application

The app includes:

- Modern dark UI design  
- Custom CSS styling  
- Real-time message classification  
- Clean and professional layout  
- Sidebar project explanation  

---

# 📂 Project Structure

```
spam-detector/
│
├── app.py
├── vectorizer.pkl
├── model.pkl
├── spam.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation & Setup (Local)

## 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/spam-detector.git
cd spam-detector
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run the App

```bash
streamlit run app.py
```

---

# ☁️ Deployment (Streamlit Cloud - Free)

1. Push project to GitHub  
2. Go to https://share.streamlit.io  
3. Connect your repository  
4. Select `app.py`  
5. Click Deploy  

Your app will be live in minutes 🚀

---

# 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- NLTK  
- Scikit-learn  
- TF-IDF Vectorizer  
- Multinomial Naive Bayes  
- Streamlit  
- Pickle  

---

# 🎯 Key Features

✔ NLP-based preprocessing  
✔ TF-IDF feature extraction  
✔ Model comparison & evaluation  
✔ Clean Streamlit UI  
✔ Real-time spam prediction  
✔ Deployable on free cloud platform  

---

# 👨‍💻 Author

**Shahrier Sabit**  
BSc in Computer Science & Engineering  
Aspiring Machine Learning Engineer  

---

# ⭐ If You Like This Project

Give it a ⭐ on GitHub and feel free to fork or improve it!
