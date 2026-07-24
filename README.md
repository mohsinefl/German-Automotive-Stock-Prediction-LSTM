# German Automotive Stock Analysis and Price Prediction using LSTM

## Exploratory Data Analysis, Technical Indicators and Deep Learning


![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green)
![Time Series](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-purple)


# Introduction

Financial time series forecasting is one of the most challenging tasks in machine learning due to market volatility, nonlinear behaviour and the influence of external economic factors.

This project investigates whether **Long Short-Term Memory (LSTM)** neural networks can learn temporal patterns from historical stock data and predict future closing prices of major German automotive companies.

The project combines:

- Exploratory Data Analysis (EDA)
- Financial time series analysis
- Technical indicators
- Feature engineering
- Deep learning
- Model evaluation
- Out-of-sample validation on unseen market data


Companies analyzed:

- BMW AG
- Mercedes-Benz Group
- Volkswagen AG


The main objective is to demonstrate a complete end-to-end Data Science workflow from raw financial data collection to deep learning-based forecasting.


---

# Business Objective

The goal of this project is to evaluate whether historical stock prices combined with technical indicators can improve the prediction of future stock prices.

The analysis focuses on:

- Understanding market behaviour
- Identifying relationships between automotive stocks
- Engineering meaningful financial features
- Developing an LSTM forecasting model
- Evaluating model performance on unseen data


The project is **not intended as a trading system or financial advice**, but as a machine learning application for sequential data.


---

# Dataset

Historical stock market data was collected using the **Yahoo Finance API**.

## Companies

| Company | Ticker |
|---|---|
| BMW AG | BMW.DE |
| Mercedes-Benz Group | MBG.DE |
| Volkswagen AG | VOW3.DE |


## Historical Period

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Close</th>
      <th>High</th>
      <th>Low</th>
      <th>Open</th>
      <th>Volume</th>
      <th>Company</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-01-02</th>
      <td>76.742256</td>
      <td>78.195325</td>
      <td>75.829867</td>
      <td>78.195325</td>
      <td>1116700</td>
      <td>Volkswagen</td>
    </tr>
    <tr>
      <th>2019-01-03</th>
      <td>75.897438</td>
      <td>76.956265</td>
      <td>75.514463</td>
      <td>76.032610</td>
      <td>968713</td>
      <td>Volkswagen</td>
    </tr>
    <tr>
      <th>2019-01-04</th>
      <td>79.118973</td>
      <td>79.118973</td>
      <td>76.505704</td>
      <td>76.618343</td>
      <td>1177680</td>
      <td>Volkswagen</td>
    </tr>
    <tr>
      <th>2019-01-07</th>
      <td>79.209099</td>
      <td>80.222869</td>
      <td>78.961287</td>
      <td>79.918734</td>
      <td>936596</td>
      <td>Volkswagen</td>
    </tr>
    <tr>
      <th>2019-01-08</th>
      <td>80.538261</td>
      <td>82.239142</td>
      <td>77.924992</td>
      <td>78.285443</td>
      <td>1517810</td>
      <td>Volkswagen</td>
    </tr>
  </tbody>
</table>
</div>

```


The dataset contains:

- Open price
- High price
- Low price
- Closing price
- Adjusted closing price
- Trading volume


For final model validation:

```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Actual Close</th>
      <th>Predicted Close</th>
      <th>Absolute Error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2026-02-06</td>
      <td>83.617996</td>
      <td>84.363930</td>
      <td>0.745934</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2026-02-09</td>
      <td>83.428894</td>
      <td>85.404518</td>
      <td>1.975624</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2026-02-10</td>
      <td>84.412193</td>
      <td>85.388634</td>
      <td>0.976440</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2026-02-11</td>
      <td>84.582375</td>
      <td>85.083618</td>
      <td>0.501244</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2026-02-12</td>
      <td>82.937248</td>
      <td>84.519783</td>
      <td>1.582535</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2026-02-13</td>
      <td>84.052910</td>
      <td>83.465164</td>
      <td>0.587746</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2026-02-16</td>
      <td>84.601288</td>
      <td>83.596436</td>
      <td>1.004852</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2026-02-17</td>
      <td>85.622398</td>
      <td>84.419838</td>
      <td>1.202560</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2026-02-18</td>
      <td>86.794800</td>
      <td>85.161987</td>
      <td>1.632812</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2026-02-19</td>
      <td>85.168579</td>
      <td>85.838387</td>
      <td>0.669807</td>
    </tr>
  </tbody>
</table>
</div>


```

completely unseen BMW stock data was used.


---

# Project Workflow


```
Data Collection
        |
        ↓
Data Cleaning
        |
        ↓
Exploratory Data Analysis
        |
        ↓
Feature Engineering
        |
        ↓
Data Normalization
        |
        ↓
Sequence Generation
        |
        ↓
LSTM Model Training
        |
        ↓
Model Evaluation
        |
        ↓
Out-of-Sample Validation
```


---

# Exploratory Data Analysis

Before building the neural network, extensive exploratory analysis was performed.

## Price Analysis

Implemented:

- Closing price comparison
- Long-term trend analysis
- Moving averages


Moving averages:

- MA 10
- MA 20
- MA 50


## Return Analysis

Daily returns were analysed to understand:

- Stock volatility
- Return distributions
- Extreme market movements


## Correlation Analysis

The relationship between automotive companies was investigated using:

- Correlation matrices
- Heatmaps
- Pairplots
- Scatter plots


### Key Findings

- BMW and Mercedes-Benz show the strongest relationship.
- Volkswagen demonstrates higher volatility.
- Automotive stocks are influenced by similar market conditions.


---

# Feature Engineering

To provide additional information to the neural network, technical indicators were created.

Implemented features:


| Feature | Description |
|-|-|
| Close Price | Historical closing price |
| SMA 20 | Short-term trend indicator |
| SMA 50 | Long-term trend indicator |
| RSI 14 | Momentum indicator |


These indicators help the model capture:

- Short-term movements
- Long-term trends
- Market momentum


All features are calculated only from historical data to avoid data leakage.


---

# Data Preprocessing

Neural networks perform better when input variables are normalized.

Therefore, Min-Max Scaling was applied:


```
Feature values → [0,1]
```


The scaler is fitted only on training data and reused for future predictions:

```python
scaler.fit_transform()
```

Training:

```python
scaler.transform()
```

Future unseen data.


---

# Sequence Generation

LSTM networks require sequential input data.

A sliding window approach was used.


Parameter:

```python
WINDOW_SIZE = 60
```


Meaning:

The model receives the previous **60 trading days** to predict the next observation.


Input structure:


```
(60 trading days × features)
```


Example:

```
Day -60
   |
Day -59
   |
 ...
   |
Day -1

   ↓

LSTM Network

   ↓

Predicted Closing Price
```


---

# Dataset Split


The data was divided into:


| Dataset | Purpose |
|-|-|
| Training | Model learning |
| Testing | Performance evaluation |
| Validation | Final model selection |


Validation consists of the last 30 observations before unseen 2026 data.


---

# LSTM Neural Network Architecture


The model uses a stacked LSTM architecture:


```
Input Sequence

(60 days × features)

        ↓

LSTM Layer
64 Units
return_sequences=True

        ↓

Dropout
0.3

        ↓

LSTM Layer
64 Units
return_sequences=True

        ↓

Dropout
0.3

        ↓

LSTM Layer
64 Units

        ↓

Dropout
0.3

        ↓

Dense Layer

        ↓

Predicted Closing Price
```


## Model Configuration


| Parameter | Value |
|-|-|
| Architecture | Stacked LSTM |
| LSTM Layers | 3 |
| Units per Layer | 64 |
| Dropout | 0.30 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |
| Metric | RMSE |
| Epochs | 100 |
| Batch Size | 32 |


---

# Model Training


To reduce overfitting:

**Early Stopping** was implemented.


Configuration:


```
monitor = val_loss

patience = 10

restore_best_weights = True
```


20% of the training data was used as validation during training.


---

# Model Evaluation


The model was evaluated using:


## Metrics

- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)


Visual evaluations:

- Training vs Validation Loss
- Prediction vs Actual values
- Prediction error analysis


---

# Prediction Results


The trained model was evaluated on:

- Training data
- Testing data
- Validation data


The prediction curves show that the LSTM model can capture general market trends while deviations occur during highly volatile periods.


Example:

```
images/prediction_vs_actual.png
```


---

# Out-of-Sample Validation (2026)


A final validation experiment was performed using completely unseen BMW stock data.


Period:

```
01.01.2026 - Present
```


The workflow:


1. Download new BMW market data

2. Calculate identical technical indicators:

   - SMA 20
   - SMA 50
   - RSI


3. Apply the previously fitted scaler


4. Generate 60-day sequences


5. Predict using the trained LSTM model


6. Compare:

- Actual closing price
- Predicted closing price
- Prediction error


Results are stored as:


```
results/predictions_2026.csv
```


Example:


| Date | Actual Close | Predicted Close | Error |
|-|-|-|-|
|2026-01-02|XX.XX|XX.XX|XX.XX|


---

# Technologies


## Programming

- Python


## Data Analysis

- Pandas
- NumPy


## Visualization

- Matplotlib
- Seaborn
- Plotly


## Machine Learning

- Scikit-Learn


## Deep Learning

- TensorFlow
- Keras


## Financial Data

- Yahoo Finance API
- yfinance


---

# Repository Structure


```
German-Automotive-Stock-Prediction-LSTM/

│
├── notebooks/
│   └── German_Automotive_Stock_Prediction_LSTM.ipynb
│
├── images/
│   ├── architecture.png
│   ├── training_history.png
│   └── prediction_vs_actual.png
│
├── results/
│   └── predictions_2026.csv
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```


---

# Installation


Clone repository:


```bash
git clone https://github.com/USERNAME/German-Automotive-Stock-Prediction-LSTM.git
```


Install dependencies:


```bash
pip install -r requirements.txt
```


Run notebook:


```bash
jupyter notebook
```


---

# Future Improvements


Possible extensions:


- MACD indicator integration
- Bollinger Bands
- ATR volatility indicator
- Candlestick pattern analysis
- Hyperparameter optimization
- Bidirectional LSTM
- GRU comparison
- Transformer-based forecasting
- Financial news sentiment analysis
- Explainable AI methods


---

# Disclaimer


This project is created for educational and portfolio purposes.

Stock markets are highly unpredictable. The model should not be considered financial advice or a trading strategy.


---

# Author


**Mohsine Falih**

Master Data Science


GitHub:

https://github.com/mohsinefl