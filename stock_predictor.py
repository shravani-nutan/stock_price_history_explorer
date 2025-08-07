import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Load trained model
model = load_model("stock_prediction_model.keras")


# Streamlit UI
st.header("Stock Price History Explorer")


# User input
stock = st.text_input("Enter stock symbol", 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download data
data = yf.download(stock, start, end)

st.subheader('Stock data')
st.write(data)

# Prepare train/test data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scaler = scaler.fit_transform(data_test)

# Prepare test sequences
x = []
y = []

for i in range(100, data_test_scaler.shape[0]):
    x.append(data_test_scaler[i-100:i])
    y.append(data_test_scaler[i, 0])

x = np.array(x)
y = np.array(y)

# You can now use x for model.predict(x), etc.
# For example:
# predicted = model.predict(x)
# st.line_chart(predicted)

