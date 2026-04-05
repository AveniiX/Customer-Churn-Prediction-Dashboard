# Customer Churn Prediction Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red)
![AUC](https://img.shields.io/badge/ROC--AUC-0.84-green)

A machine learning pipeline that predicts customer churn
for a telecom company, deployed as an interactive business
dashboard with real-time predictions and data visualizations.

**[Live Demo](https://customer-churn-prediction-dashboard-u7avckzktypqdb3i5fx6jy.streamlit.app/)**

---

## Overview

Built an end-to-end churn prediction system:
raw customer data → preprocessing → SMOTE balancing
→ Random Forest classifier → interactive Streamlit dashboard.

**Dataset**: IBM Telco Customer Churn (Kaggle)
- 7,043 customers, 20 features
- Imbalanced: 73% No Churn / 27% Churn

**Model**: Random Forest with SMOTE oversampling
- ROC-AUC: ~0.84
- Handles class imbalance via SMOTE

---

## Key Business Insights

- Month-to-month contract customers churn 3x more than
  yearly contract customers
- Customers with tenure under 12 months are highest risk
- Fiber optic internet users churn more than DSL users
- High monthly charges correlate strongly with churn

---

## Results

| Model | Accuracy | ROC-AUC | Recall (Churn) |
|-------|----------|---------|----------------|
| Random Forest + SMOTE | 79.1% | 0.84 | 0.81 |
| Logistic Regression | 76.3% | 0.81 | 0.78 |

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn — Random Forest, Logistic Regression
- imbalanced-learn — SMOTE oversampling
- Plotly + Streamlit — interactive dashboard
- Pickle — model serialization

---

## How to Run

```bash
git clone https://github.com/AveniiX/churn-prediction
cd churn-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Author

**Muhammad Hamam** — Data Scientist
[LinkedIn](https://www.linkedin.com/in/muhammad-hamam-yousif-b90455374/) | [GitHub](https://github.com/AveniiX)
