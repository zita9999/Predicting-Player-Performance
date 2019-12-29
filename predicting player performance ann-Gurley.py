# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#importing the dataset
dataset = pd.read_csv('todd-gurley stats.csv')
training_set = dataset.iloc[:44, :].values
X = training_set[:, 8:28]
y = training_set[:,28]

#splitting the dataset into the training and test set
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.0001, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing the Keras library and making the ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

regressor = Sequential()

#adding the input layer and the first hidden layer with dropout
regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu', input_dim =14))
regressor.add(Dropout(p = 0.01))

#adding the second hidden layer
regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu'))
regressor.add(Dropout(p = 0.01))

#adding a thrid hidden layer
regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu'))
regressor.add(Dropout(p = 0.01))

#adding a fourth hidden layer
regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu'))
regressor.add(Dropout(p = 0.01))


#adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'normal',activation = 'linear'))

#compiling the ANN
regressor.compile(optimizer = 'rmsprop', loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])

#Fitting the ANN to the training set
regressor.fit(X_train, y_train, batch_size = 4, nb_epoch = 500)


new_pred1 = regressor.predict(sc.transform(np.array([[17.7,299.7,4.8,61.3,38.4,219.7,1.375,3.125,72.9,21.2,80,0.3125,4,2,32.9,421.1,281.7,2,139.4,1.4375]])))

#predicting the fantasy points for the 2018 season
test = dataset.iloc[44:58,8:28].values
test = sc.transform(test)
predicted_fantasy = regressor.predict(test)
predicted_fantasy = predicted_fantasy[:,0]
real_fantasy = dataset.iloc[44:58,28].values