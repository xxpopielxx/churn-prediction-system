# 🛒 End-to-End Customer Churn Prediction Pipeline

A comprehensive Machine Learning pipeline designed to predict customer churn. This project demonstrates the full ML lifecycle: from raw data analysis and processing to model training and cloud deployment.

## 🎯 Business Goal

Customer retention is significantly cheaper than acquisition. The goal of this system is to identify customers with a high risk of attrition (churn) based on their demographics and service usage. This allows the business to take proactive retention actions (e.g., offering discounts) to keep valuable customers.

## 🏗️ Pipeline Architecture

The project follows a structured 4-step workflow:

1.  **Data Engineering & EDA:** Data cleaning, handling missing values, and Feature Engineering (One-Hot Encoding). Deep dive into data distribution.
2.  **Model Training:** Training and evaluating ML models (Logistic Regression, Random Forest, LightGBM) focusing on business metrics like Recall and F1-Score (handling class imbalance).
3.  **Deployment (API):** Serving the trained model as a REST API using **FastAPI**.
4.  **Infrastructure (Ops):** Containerizing the application with **Docker** and deploying it to the cloud (**AWS EC2**).

## 🛠️ Tech Stack

- **Language:** Python 3.12
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, LightGBM / Random Forest
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** FastAPI, Uvicorn, Docker
- **Cloud:** AWS (EC2)

## 📂 Project Structure

```text
churn-prediction-system/
├── data/              # Raw and processed datasets (ignored by Git)
├── notebooks/         # Jupyter Notebooks for EDA and experiments
├── src/               # Source code for training and API
├── venv/              # Virtual Environment
├── .gitignore         # Files to ignore (e.g., large data files)
└── README.md          # Project documentation
```
