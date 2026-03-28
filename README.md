# ecommerce-analytics
# 🛒 E-commerce Analytics Dashboard

## 📌 Overview

This project develops a comprehensive data analytics and recommendation system for an e-commerce platform using the Olist dataset.

The system integrates machine learning, customer segmentation, and recommendation techniques into an interactive Streamlit dashboard, allowing users to explore business insights, predict customer behavior, and receive product recommendations.

---

## 📊 Dataset

* Source: Olist E-commerce Dataset (Brazil)
* Processed files:

  * `cleaned_data_small.csv`
  * `rfm_data.csv`
  * `rules.csv` (Association Rules)

---

## ⚙️ Key Features

### 📊 1. Dashboard (Business Overview)

* Total Orders, Customers, Revenue
* Top product categories by revenue
* Order trends over time (time series)

👉 Purpose:
Provide a quick overview of business performance

---

### 👥 2. Customer Segmentation (RFM)

* Uses:

  * Recency
  * Frequency
  * Monetary
* Clustering algorithm: **KMeans**
* Interactive cluster selection (k = 2 → 8)

👉 Output:

* Visual customer segments (scatter plot)

👉 Insight:
Identify high-value customers and potential churn groups

---

### 🎯 3. Recommendation System

* Input: Customer ID
* Logic:

  * Existing user → personalized recommendation
  * New user → popular products

👉 Output:

* Top recommended products
* Displayed as interactive product cards

👉 Purpose:
Improve customer experience and increase sales

---

### 🛍️ 4. Market Basket Analysis (FP-Growth)

* Displays association rules from `rules.csv`
* Sorted by lift

👉 Metrics:

* Support
* Confidence
* Lift

👉 Purpose:
Discover product relationships for cross-selling strategies

---

### 🔮 5. Prediction (Review Score)

* Model: Machine Learning (saved as `classifier.pkl`)
* Input:

  * Price
  * Freight value
  * Payment value

👉 Output:

* Predicted review score

👉 Purpose:
Estimate customer satisfaction

---

### ⚙️ 6. Admin Panel

* Upload new dataset
* Retrain model (Random Forest)

👉 Purpose:
Enable dynamic updates and model retraining

---

## 🧠 Technologies Used

* Python (Pandas, NumPy)
* Scikit-learn
* Plotly (Visualization)
* Streamlit (Web App)
* Joblib (Model saving/loading)

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py
├── cleaned_data_small.csv
├── rfm_data.csv
├── rules.csv
├── classifier.pkl
└── README.md
```

---

## 📈 Key Insights

* Customer behavior is highly segmented based on RFM metrics.
* High-value customers contribute significantly to revenue.
* Product recommendations improve personalization.
* Association rules reveal strong co-purchase patterns.
* Review scores can be predicted based on transaction features.

---

## 🌐 Demo

👉 Streamlit App: *https://ecommerce-analytics-bigdata.streamlit.app/*
👉 GitHub Repo: *(add your link here)*

---

## 📌 Conclusion

This project demonstrates how data analytics and machine learning can be integrated into a real-world e-commerce system to generate actionable insights and intelligent recommendations.

---

## 👨‍💻 Author

* Nhóm 10_BIG DATA
* Course: Big Data Analytics
* University: HCMUTE

---
