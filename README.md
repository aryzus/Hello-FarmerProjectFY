# Hello-FarmerProjectFY
Developed an AI‑powered system for crop recommendation, plant disease detection, weather forecasting, and mandi price prediction using Python and ML models, deployed via GitHub to support farmers with data‑driven insights.



# 🌾 Mandi Price Prediction Under Various Climatic Conditions

A machine learning-based system to predict agricultural mandi prices using time-series data and climatic factors such as temperature, rainfall, and humidity.

---

## 📌 Project Overview

This project aims to forecast mandi prices (specifically onion prices) by analyzing historical market data along with climatic conditions. The system leverages machine learning models to capture price trends and enhance prediction accuracy.

---

## 🎯 Objectives

- Predict mandi prices using historical data  
- Incorporate climatic variables into prediction  
- Analyze impact of weather on price trends  
- Build an interactive visualization dashboard  

---

## 🧠 Features

- 📊 Time-series analysis of mandi prices  
- 🌦️ Integration of climatic features (temperature, rainfall, humidity)  
- ⚙️ Feature engineering (lag values, rolling mean, volatility)  
- 🤖 Machine learning model (XGBoost / Random Forest)  
- 📈 Visualization of actual vs predicted prices  
- 💡 Explainable AI using SHAP  

---

## 🗂️ Dataset

- Source: Agricultural mandi dataset (CSV)  
- Commodity: Onion  
- Market: Lasalgaon (or selected market)  
- Features:
  - Modal Price  
  - Date  
  - Market details  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Data cleaning and formatting  
- Date conversion  
- Filtering specific commodity and market  

### 2. Feature Engineering
- Lag features (lag_1, lag_7)  
- Rolling mean and standard deviation  
- Climate features (temperature, rainfall, humidity)  

### 3. Model Training
- XGBoost Regressor / Random Forest  
- Time-based train-test split  

### 4. Evaluation
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Square Error)  

---

## 📊 Results

- Model successfully predicts price trends  
- Predictions closely follow actual prices  
- SHAP analysis shows:
  - Lag features are most influential  
  - Climate features have moderate impact  

---

## 🖥️ Tech Stack

- Python 🐍  
- Pandas & NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib  
- Streamlit  

---

## 🚀 How to Run

```bash
# Clone repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to project
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
