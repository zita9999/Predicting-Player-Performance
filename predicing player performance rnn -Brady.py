# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Part 1 - Data preprocessing
#keras neural networks only works for arrays
dataset = pd.read_csv('file:///C:/Users/chris/OneDrive/Documents/Courses and Projects/Predicitng player performance/Tom-Brady stats copy.csv')
dataset = dataset.iloc[:107,:]
training_set = dataset.iloc[:91,23:24].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scale.fit_transform(training_set)

#splitting it into the training set
X_train = []
y_train = []
for i in range(8,91):
    X_train.append(training_set_scaled[i-8:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1 ))

#importing the kereas library and making the rnn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#Adding the first LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding a second LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


#Adding a thrid LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


#Adding a fourth LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

#adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 100, batch_size = 4)


dataset_test = dataset.iloc[:91,23:24].values
inputs = dataset_test[len(dataset_test) - 9:]
inputs = inputs.reshape(-1,1)

#since the rnn was trained on the scaled data we also scale the test data than
inputs = scale.transform(inputs)

X_test = []
for i in range(8,9):
    X_test.append(inputs[i-8:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1 ))

#predicitng the next games projected fantasy points
predict = regressor.predict(X_test)
predict = scale.inverse_transform(predict)






