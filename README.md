# Amazon Stock Forecast RNN(LSTM, GRU, Simple RNN)

## Introduction
This project aims to perform forecasting of Amazon (AMZN) stock prices using a Recurrent Neural Network (RNN)-based approach, including the following models:

- Simple RNN

- Long Short-Term Memory (LSTM)

- Gated Recurrent Unit (GRU)

The primary focus of the prediction is on daily closing price data (Close), which is commonly used in technical analysis of stock markets.

---

## Technology
- **Data Manipulation**: `numpy`, `pandas`
- **Data Visualization**: `matplotlib`
- **Model Training**: `pytorch`
- **Model Evaluation**: `scikit-learn`

---

## Set up Environment
1. Clone this repository by entering the following command:
```
git clone https://github.com/PutraAlFarizi15/Amazon-Stock-Forecast-Simple-RNN-LSTM-GRU.git
```
2. Once the git clone process is complete, navigate to the project folder:
```
cd Amazon-Stock-Forecast-Simple-RNN-LSTM-GRU
```
3. Create a virtual environment inside the project folder:
```
python -m venv venv
```
4. Activate the virtual environment using the following command:
```
venv\scripts\activate
```
5. If the virtual environment is activated, install the required libraries using the requirements.txt file:
```
pip install -r requirements.txt
```
---
## Usage
Open the **research.ipynb** file and run each cell in the notebook. This file contains the analysis results from the model training process, including performance evaluation and visualizations of the data and metrics used.

---

# 📊 Analysis
---

## Dataset Overview

![Amazon stock price](images/Closed%20Amazon.png)

The AMZN stock remained below $5 until 2009, began increasing steadily to $25 by 2016, and surged to $180 during the COVID-19 pandemic before correcting to under $100 in 2023.

---

## Data Splitting
To train and evaluate the models properly, we split the dataset as follows:

- 80% Training Data: Used to train the model.

- 20% Testing Data: Used to evaluate the model’s generalization ability on unseen data.

From the 80% training data:

- We further divide it using TimeSeriesSplit Cross Validation, which respects the temporal order of the data to prevent data leakage.

- In each split, earlier data is used for training and later data for validation.

- This approach simulates a real-world scenario where future data should never influence past predictions.

This ensures a robust evaluation while maintaining the integrity of time series modeling.

---

## Model Architectures
- Simple RNN: Basic RNN layer; suffers from vanishing gradient in long sequences.

- LSTM: Uses input, forget, and output gates to retain long-term dependencies.

- GRU: Simplified version of LSTM with update and reset gates; often faster and equally effective.

---

## Evaluation Metrics
- RMSE: Penalizes large errors more heavily.

- MAE: Measures average absolute error.

- MAPE: Measures percentage error, useful for cross-scale comparison.

---

## 📉 Loss Comparison
![Graph Loss](images/Graph%20Loss.png)
All models show comparable loss values. However, deeper evaluation requires considering RMSE, MAE, and MAPE

---

## 📊 Model Performance
![Comparation Model](images/Comparation%20Model.png)

GRU outperforms other models in all three metrics, indicating it generalizes best to the test data.

---

## 🔁 Cross Validation (TimeSeriesSplit)
Traditional K-Fold isn't ideal for time series due to data leakage. We use TimeSeriesSplit, which respects chronological order.

![Comparation Model CV](images/Comparation%20Model%20CV.png)

GRU again shows the best performance on both training and validation sets, reinforcing its robustness.

---

## ✅ Final Evaluation
![Actual vs Predicted](images/Actual%20vs%20Predicted%20GRU.png)

The GRU model closely follows the actual values in the test set, demonstrating its effectiveness in capturing temporal patterns.

---

## 📚 Conclusion
GRU delivers the best forecasting performance for AMZON stock prices among the RNN-based models tested. This solution highlights the importance of model selection and evaluation strategy in time series forecasting.

---

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)