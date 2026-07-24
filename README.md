# German Automotive Stock Analysis and Price Prediction using LSTM

## Deep Learning for Time Series Forecasting with Technical Indicators

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Machine Learning](https://img.shields.io/badge/Domain-Machine%20Learning-green)
![Time Series](https://img.shields.io/badge/Task-Time%20Series%20Forecasting-purple)


## Overview

This project demonstrates a complete end-to-end Data Science workflow for predicting stock prices of major German automotive companies using Long Short-Term Memory (LSTM) neural networks.

The project combines:

- Exploratory Data Analysis (EDA)
- Financial time-series analysis
- Technical indicator engineering
- Data preprocessing
- Deep learning modeling
- Model evaluation
- Out-of-sample validation on unseen market data


The objective is to build a reproducible machine learning pipeline that predicts future closing prices based on historical market behavior.


---

# Business Objective

Financial markets are highly dynamic and influenced by many external factors. 

The goal of this project is not to create a trading strategy, but to demonstrate how deep learning methods can be applied to sequential financial data.

The project addresses the following question:

> Can an LSTM neural network learn temporal patterns from historical stock prices and generate meaningful predictions on unseen market data?


---

# Dataset

Historical stock market data was collected using the Yahoo Finance API.

Companies analyzed:

| Company | Ticker |
|---|---|
| BMW AG | BMW.DE |
| Mercedes-Benz Group | MBG.DE |
| Volkswagen AG | VOW3.DE |


Dataset period:

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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