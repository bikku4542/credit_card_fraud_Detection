# 💳 Credit Card Fraud Detection

A machine learning project aimed at detecting fraudulent credit card transactions using various classification algorithms including Logistic Regression, Support Vector Machine (SVM), and Random Forest. This project also includes exploratory data analysis and visualizations to understand the distribution and patterns in the data.


## 🚀 Project Objective

To build a predictive model that can accurately identify fraudulent credit card transactions from a real-world dataset. The project focuses on improving detection accuracy by comparing multiple machine learning models and evaluating their performance.

## 🧠 Problem Statement

Credit card fraud is a growing concern worldwide, leading to significant financial losses. The goal is to develop a system that detects fraudulent transactions with high precision while minimizing false positives.
## 📁 Dataset

- **Source**: [kaggale dataset-  "C:\Users\Dell\Downloads\creditcard.csv"])
- **Description**: Contains transactions made by European cardholders in September 2013.
- **Size**: 284,807 transactions with 492 frauds (highly imbalanced).
- **Features**:
  - Numerical input variables (V1 to V28 - PCA transformed)
  - `Amount` – Transaction Amount
  - `Class` – Target variable (0 = Non-Fraud, 1 = Fraud)

## 🛠️ Tools & Technologies

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, SVM, Random Forest)


## 📊 Exploratory Data Analysis (EDA)

- Visualized class imbalance using pie and bar charts
- Distribution of transaction amounts and time
- Correlation heatmap to observe feature relationships
- Histogram plots to identify data skewness


## 🧪 Machine Learning Models

### 1. Logistic Regression
- Fast and interpretable model
- Used as a baseline

### 2. Support Vector Machine (SVM)
- Tested with linear kernel
- Good at handling small, imbalanced datasets

### 3. Random Forest
- Ensemble model with multiple decision trees
- Often improves accuracy and reduces overfitting

## 📈 Model Evaluation

- Used metrics: Accuracy Score, Confusion Matrix
- Evaluated each model on training and test sets
- Accuracy Comparison:

| Model               | Accuracy (Test Set) |
|--------------------|---------------------|
| Logistic Regression| 99.3%+              |
| SVM                | 99.4%+              |
| Random Forest      | 99.7%+              |

(Note: Values may vary slightly based on random train-test split.)


## 📉 Challenges Faced

- Imbalanced dataset: only 0.17% fraud cases
- Risk of overfitting due to limited fraud data
- Need for precision–recall tradeoff tuning



