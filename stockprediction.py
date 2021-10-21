"""
Stock Prediction Program
Author: Enoc Mena
Version: 1.0.0
Description: The purpose of this program is to predict ADA-CARDANO stock price by training
and using an LSTM Neural network
"""
import matplotlib
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from aiohttp import web
from pasta.augment import inline

matplotlib
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("ADA-USD-2.csv")  # import data from .csv file
data.head()
data["Date"] = pd.to_datetime(data.Date, format="%Y-%m-%d")  # format the date to follow [year, month, day]
data.index = data['Date']

plt.figure(figsize=(16, 8))  # 16 x 8 inch data frame
plt.plot(data["Close"], label='Close Price history')
plt.title('Ada-Stock-Chart')

dataset = data.sort_index(ascending=True, axis=0)
new_dataset = pd.DataFrame(index=range(0, len(data)), columns=['Date', 'Close'])
for i in range(0, len(data)):
    new_dataset["Date"][i] = dataset['Date'][i]
    new_dataset["Close"][i] = dataset["Close"][i]

# normalize new data set
scale = MinMaxScaler(feature_range=(0, 1))
final_dataset = new_dataset.values

trainData = final_dataset[0:987, :]
validData = final_dataset[987:, :]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

scale = MinMaxScaler(feature_range=(0, 1))
scale_data = scale.fit_transform(final_dataset)

# append the data into the arrays
x_train_data, y_train_data = [], []
for i in range(60, len(trainData)):
    x_train_data.append(scale_data[i - 60: i, 0])
    y_train_data.append(scale_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = x_train_data.reshape(x_train_data, x_train_data.shape[0], x_train_data.shape[1], 1)

# LSTM model
lstm = Sequential()
lstm.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1, 1])))
lstm.add(LSTM(units=50))
lstm.add(Dense(1))

data_input = new_dataset[len(new_dataset) - len(validData) - 60:].values
data_input.data_input.reshape(-1, 1)
data_input = scale.transform(data_input)

lstm.compile(loss='mean_squared_error', optimizer='adam')
lstm.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

# sample of data set
x_sample_test = []
for i in range(60, data_input.shape[0]):
    x_sample_test.append(data_input[i - 60: i, 0])
x_sample_test = np.array(x_sample_test)
x_sample_test = np.reshape(x_sample_test, (x_sample_test.shape[0], x_sample_test.shape[1], 1))

closing_prediction_price = lstm.predict(x_sample_test)
closing_prediction_price = scale.inverse_transform(closing_prediction_price)

lstm.save("ada_stock.h5")  # save the model

trainData = new_dataset[:987]
validData = new_dataset[987:]
validData['Predictions'] = closing_prediction_price

plt.plot(trainData["Close"])
plt.plot(validData[['Close', "Predictions"]])
plt.show()
