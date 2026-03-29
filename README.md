# Telco Customer Churn Prediction

## Overview
This project predicts customer churn for a telecom company using machine learning. It analyzes customer demographics, services, and billing data to identify customers who are likely to leave.

The project is implemented in Google Colab and follows a step-by-step machine learning pipeline from data understanding to model building.

---

## Dataset
The dataset used is the **Telco Customer Churn Dataset** from Kaggle.

Dataset:    
https://www.google.com/url?q=https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fblastchar%2Ftelco-customer-churn

It contains information about:
- Customer demographics (gender, senior citizen, etc.)
- Account details (tenure, contract type)
- Services (internet, phone, streaming)
- Billing (monthly and total charges)

**Target Variable:** `Churn`  
- Yes → Customer left  
- No → Customer stayed  

---

## Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Project Workflow

### 1. Data Loading
- Dataset loaded using Pandas
- Initial inspection using `.head()`, `.shape()`, `.columns()`

---

### 2. Data Understanding
- Checked distribution of target variable (`Churn`)
- Identified:
  - Numerical features → tenure, MonthlyCharges, TotalCharges  
  - Categorical features → gender, contract, services  

---

### 3. Data Cleaning
- `TotalCharges` converted from object to numeric  
- Invalid values converted to NaN using `errors='coerce'`  
- Missing values filled using **median imputation**  
- Dropped `customerID` (not useful for prediction)

---

### 4. Feature & Target Split
- Features (`X`) → all columns except Churn  
- Target (`y`) → Churn column  

---

### 5. Encoding
- Target variable encoded:
  - Yes → 1  
  - No → 0  

- Categorical features encoded using:
  - One-Hot Encoding (`pd.get_dummies`)
  - `drop_first=True` to avoid multicollinearity  

---

### 6. Model Preparation
- Dataset prepared for machine learning after encoding  
- Feature dimensions increased after encoding  

---

### 7. Model Building & Evaluation
- Machine learning model trained (e.g., Logistic Regression / Random Forest)  
- Model evaluated using:
  - Accuracy  
  - Confusion Matrix  
  - Classification Report  

---

## Key Insights
- Customers with higher monthly charges are more likely to churn  
- Contract type plays a major role in churn behavior  
- Long-term customers are less likely to leave  
