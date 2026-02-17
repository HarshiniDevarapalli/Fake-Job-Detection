# Fake Job Posting Detection using NLP and Machine Learning

An end-to-end Machine Learning and NLP system that detects fraudulent job postings by analyzing job descriptions and company credibility signals. The system provides a **fake probability score** and classifies postings as **Legitimate, Suspicious, or Fake** through an interactive web interface.

---

## Demo

Users can input job details or job posting links, and the system predicts fraud probability in real-time.

---

## Features

* Fraud detection using **Natural Language Processing**
* **Probability score** indicating likelihood of fraud
* Domain credibility analysis:

  * Website availability check
  * Free vs corporate email detection
  * Company-domain consistency validation
  * Domain suffix analysis
* Interactive and user-friendly web interface
* Real-time prediction using trained ML model

---

## Machine Learning Pipeline

**Dataset:** Kaggle Fake Job Postings Dataset

**Steps:**

1. Data Cleaning and Preprocessing
2. Feature Engineering

   * TF-IDF text features
   * Domain credibility features
3. Model Training using Random Forest Classifier
4. Model Evaluation
5. Model Deployment using Joblib

---

## Tech Stack

**Languages & Libraries**

* Python
* Scikit-learn
* Pandas
* NumPy
* Flask
* Joblib
* TF-IDF Vectorizer
* Random Forest
* tldextract
* Requests

---

## Project Structure

```
Fake-Job-Detection/

model/
   fake_job_model_v3.pkl

src/
   app.py
   train_model.py

screenshots/
   fake.png
   suspicious.png
   legitimate.png

requirements.txt
README.md
```
---

## How it Works

The system combines:

* NLP-based text analysis
* Domain trust verification
* Machine learning classification

to accurately detect fraudulent job postings.

---


## Author

**Harshini Devarapalli**

GitHub: https://github.com/HarshiniDevarapalli


