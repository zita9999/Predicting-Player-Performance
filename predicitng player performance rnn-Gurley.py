# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('todd-gurley-stats.csv')
dataset = dataset.iloc[:58,:]
training_set = dataset.iloc[:57,28:29].values

#feature scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scale.fit_transform(training_set)

#getting the training set
X_train = []
y_train = []
for i in range(8,57):
    X_train.append(training_set_scaled[i-8:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1 ))

#importing the keras library and making the rnn
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

#fitting the RNN onto our training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 4)



dataset_test = dataset.iloc[:57,28:29].values
inputs = dataset_test[len(dataset_test) - 9:]
inputs = inputs.reshape(-1,1)

#since the rnn was trained on the scaled data we also scale the test data than
inputs = scale.transform(inputs)

X_test = [] 
for i in range(8,9): 
    X_test.append(inputs[i-8:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1 ))

#predicting the fantasy points for the next game
predict = regressor.predict(X_test)
predict = scale.inverse_transform(predict)

