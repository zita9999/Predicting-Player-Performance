# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('file:///C:/Users/chris/OneDrive/Documents/Courses and Projects/Tom-Brady stats copy.csv')
training_set = dataset.iloc[:,23:24].values
input_features = dataset.iloc[:,9:24].values

input_data = input_features
dataset_train = dataset.iloc[:,:].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler(feature_range = (0,1))
training_set_scaled = scale.fit_transform(training_set)
input_data = input_data[:,0:16]
input_data = scale.fit_transform(input_features[:,:])

y = input_data[:107, 14]
input_data = input_data[:, 0:14]


#multivariabel time steps
lookback= 8

test_size=int(.001 * len(dataset_train))
X=[]

for i in range(len(dataset_train)-lookback-1):
    t=[]
    for j in range(0,lookback):        
        t.append(input_data[[(i+j)], :])
    X.append(t)

X, y= np.array(X), np.array(y)
X_test = X[106:107]
X = X.reshape(X.shape[0],lookback, 14)
X_test = X_test.reshape(X_test.shape[0],lookback, 14)
print(X.shape)
print(X_test.shape)




#Building the  Multivariable RNN and importing the keras library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

#Adding a first LSTM layer
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1],14)))
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
regressor.fit(X,y, epochs = 100, batch_size = 2) #this takes long

y_pred = regressor.predict(X_test)
a = y_pred[0,0]
zero = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,a]
zero = np.array(zero)
y_pred = zero.reshape(1,-1)
y_pred = scale.inverse_transform(y_pred)


#predciting Brady's fantasy points for the  2018 season
dataset_test = dataset.iloc[99:,9:24].values
input_feat = dataset.iloc[99:,9:24].values
input_data2 = input_feat
input_data2[:,0:15] = scale.transform(input_feat[:,:])
input_data2 = input_data2[:,0:14]

test=[]

for i in range(0,7):
    t=[]
    for j in range(0,lookback):        
        t.append(input_data2[[(j)], :])
    test.append(t)

test = np.array(test)
test = test.reshape(test.shape[0],lookback, 14)

p = regressor.predict(test)
p = scale.inverse_transform(p)









