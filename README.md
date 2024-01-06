
# Netflix Stock Price Prediction

Welcome to the Netflix Stock Price Prediction project. In this venture, we leverage advanced machine learning techniques, specifically the Long Short-Term Memory (LSTM) neural network, renowned for its efficiency in regression analysis and time series forecasting..


## Appendix

Additional Information

    Yahoo Finance API Documentation
    Plotly Documentation
    Seaborn Documentation
    Keras Documentation


## Introduction
The primary goal of this project is to predict Netflix stock prices using historical data, providing insights into potential future trends.
## Prerequisites

Required Python Libraries: Install the necessary libraries by running:

```Python

pip install pandas yfinance plotly seaborn matplotlib scikit-learn keras
```
Make sure to have an active internet connection for library installations.

## Dataset

The historical stock price data for Netflix is sourced from Yahoo Finance. You can download the dataset by running the following code in your Jupyter Notebook or Python script:

```python

import yfinance as yf

# Set the start and end dates for the data
start_date = "YYYY-MM-DD"
end_date = "YYYY-MM-DD"  # Use today's date or any desired end date
```
## Model Training

### Data Splitting

```python

from sklearn.model_selection import train_test_split

X = data[["Open", "High", "Low", "Volume"]]
y = data["Close"]
X = X.to_numpy()
y = y.to_numpy()
y = y.reshape(-1, 1)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
```
### LSTM Neural Network Architecture

```python

from keras.models import Sequential
from keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()
```
### Training the Model

```python

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(xtrain, ytrain, batch_size=1, epochs=30)
```
### Testing Model Predictions

```python

import numpy as np

# Example feature for prediction
features = np.array([[402.380021, 428.600104, 450.602002, 37282919]])
predicted_price = model.predict(features)
print(predicted_price)

```