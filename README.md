# 🧠 Authorship Attribution using Machine Learning

This repository contains the complete implementation of a machine learning pipeline for **authorship attribution**—the task of identifying the author of a tweet given a set of candidate Twitter users. The project was carried out in two phases: **data preparation** and **modeling & evaluation**.

---

## 📂 Repository Structure

---

## 🔍 Project Overview

**Goal:** Given a tweet, predict its author from a known group of Twitter users.

We approach this in two phases:

1. **Phase 1 - Data Preparation:**
   - Scraping and cleaning tweets.
   - Feature extraction using Bag of Words and Laplace smoothing.

2. **Phase 2 - Modeling & Evaluation:**
   - Feature engineering using both BoW and BERT-based sentence embeddings.
   - Model training using KNNs, Neural Networks, and Ensemble Methods.
   - Performance evaluation using accuracy, precision, recall, F1-score, and confusion matrices.

---

## 🧰 Tools & Libraries Used

- **Languages:** Python
- **Libraries:** `scikit-learn`, `nltk`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sentence-transformers`
- **Model:** `all-MiniLM-L6-v2` from Hugging Face for BERT embeddings
- **Techniques:** 
  - Bag of Words (BoW) with Add-1 Smoothing
  - BERT-based contextual embeddings
  - K-Nearest Neighbors (KNN)
  - Neural Networks (MLP)
  - Ensemble Learning (Bagging, Boosting, Voting)

---

## 🧾 Phase Details

### 📌 Phase 1: Data Preparation

- **Task 1:** Scraped 1000 tweets for an assigned user using the Twitter API.
- **Task 2:** Applied cleaning techniques including:
  - Lowercasing
  - Stopword removal
  - Punctuation and URL removal
  - Tokenization
- **Task 3:** 
  - Built BoW feature vectors.
  - Applied Add-1 (Laplace) smoothing to avoid zero-frequency issues.
- **Task 4:** Theoretical questions covering:
  - Preprocessing decisions
  - Classifier comparisons
  - Dimensionality discussions

---

### 📌 Phase 2: Model Implementation

- **Task 1:** 
  - Combined tweets from all group members.
  - Generated BoW and BERT-based features.
- **Task 2:** KNN Classifier
  - Applied 5-fold cross-validation.
  - Evaluated performance using multiple metrics.
- **Task 3:** Neural Network Classifier
  - Used Scikit-learn’s MLPClassifier.
  - Evaluated similarly to KNN.
- **Task 4:** Ensemble Methods
  - Explored Bagging, Boosting, and Voting ensembles.
  - Compared performance across both feature types.
- **Task 5:** Addressed theoretical questions on model behavior, scalability, and real-world applications.

---

## 📊 Results Summary

| Model       | Features | Accuracy | 
|-------------|----------|----------|
| KNN         | BoW      | 60.7%    |
| KNN         | BERT     | 81.8%    |
| NN          | BoW      | 84%      |
| NN          | BERT     | 81.6%    |
| Ensemble    | BOW      | 85%      |
| Ensemble    | BERT     | 86%      | 
| Ensemble    | BERT     | **Best** | 


---

## 💡 Key Insights

- BERT embeddings significantly outperformed traditional BoW features.
- Ensemble methods (particularly Voting Classifier) yielded the most robust performance.
- The project demonstrated how contextual understanding boosts classification accuracy.

---
