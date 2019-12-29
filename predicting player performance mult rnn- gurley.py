# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('todd-gurley-stats.csv')
training_set = dataset.iloc[:,28:29].values
input_features = dataset.iloc[:,8:29].values

input_data = input_features
dataset_train = dataset.iloc[:,:].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scale.fit_transform(training_set)
input_data = input_data[:,0:22]
input_data = scale.fit_transform(input_features[:,:])

y = input_data[:58, 20]
input_data = input_data[:, 0:20]


#Multivariable time steps
lookback= 8

test_size=int(.001 * len(dataset_train))
X=[]

for i in range(len(dataset_train)-lookback-1):
    t=[]
    for j in range(0,lookback):        
        t.append(input_data[[(i+j)], :])
    X.append(t)

X, y= np.array(X), np.array(y)
X_test = X[57:58]
X = X.reshape(X.shape[0],lookback, 20)
X_test = X_test.reshape(X_test.shape[0],lookback, 20)
print(X.shape)
print(X_test.shape)


#Building the Multivariable RNN and importing the keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#Adding a first LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1],20)))
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

#Fitting the RNN to the training set
regressor.fit(X,y, epochs = 100, batch_size = 2)

#predicitng the Gurley's Fantasy points for the 2018 season
y_pred = regressor.predict(X_test)
a = y_pred[0,0]
zero = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,a]
zero = np.array(zero)
y_pred = zero.reshape(1,-1)
y_pred = scale.inverse_transform(y_pred)
