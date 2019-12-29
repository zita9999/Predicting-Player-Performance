# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#importing the dataset and data cleaning
dataset = pd.read_csv('file:///C:/Users/chris/OneDrive/Documents/Courses and Projects/Tom-Brady stats copy.csv')
training_set = dataset.iloc[:91, :].values
X = training_set[:, 9:23]
y = training_set[:,23]


#splitting into the test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.0001, random_state = 0)


# Feature Scaling
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
regressor.fit(X_train, y_train, batch_size = 2, nb_epoch = 500)

y_pred = regressor.predict(X_test)


#predicitng a single observation

new_pred1 = regressor.predict(sc.transform(np.array([[22.5,327.2,19.8,36,54,64,300,3.25,95.2,96.1,252.8,1.6875,4,2]])))

new_pred2 = regressor.predict(sc.transform(np.array([[25,354.4,20.7,37,67,63.8,234.5,2.75,100.9,119.9,254.6,1.875,3,1]])))




#graphing predicted values and real values for toms last 8 games

test = dataset.iloc[99:107,9:23].values
test = sc.transform(test)
predicted = regressor.predict(test)
predicted = predicted[:,0]
real = dataset.iloc[99:107,5].values

n_groups = 8
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, real, bar_width, alpha = opacity, color = 'b', label = 'Real Yards')
rects1 = plt.bar(index +bar_width, predicted, bar_width, alpha = opacity, color = 'r', label = 'Predicted Fantasy Points')

plt.xlabel('Games')
plt.ylabel('Fantasy Points')
plt.title('Tom Bradys Fantasy Points')
plt.legend()

plt.tight_layout()
plt.show()


#graphing predicted real and projected fantasy points for the 2018 season
test = dataset.iloc[91:107,9:23].values
test = sc.transform(test)
predicted_fantasy = regressor.predict(test)
predicted_fantasy = predicted_fantasy[:,0]
real_fantasy = dataset.iloc[91:107,23].values
projected_fantasy = (20.2,22.8,21,18.6,21,21,22.2,20.2,16.2,0,19.8,21.2,21.8,21.4,19.6,0)

n_groups = 16
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, real_fantasy, bar_width, alpha = opacity, color = 'b', label = 'Real Fantasy Points')
rects1 = plt.bar(index +bar_width, predicted_fantasy, bar_width, alpha = opacity, color = 'r', label = 'Predicted Fantasy Points')
rects3 = plt.bar(index +bar_width, projected_fantasy, bar_width, alpha = opacity, color = 'g', label = 'Projected Fantasy Points by CBS')

plt.xlabel('Games')
plt.ylabel('Fantasy Points')
plt.title('Tom Bradys Fantasy Points')
plt.legend()

plt.tight_layout()
plt.show()


#tuning the ann
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense
def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu', input_dim =14))
    regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu'))
    regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu'))
    regressor.add(Dense(units = 7, kernel_initializer = 'normal',activation = 'relu'))
    regressor.add(Dense(units = 1, kernel_initializer = 'normal',activation = 'linear'))
    regressor.compile(optimizer = optimizer, loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])    
    return regressor

regressor = KerasRegressor(build_fn = build_regressor)

parameters = {'batch_size': [4,8],
              'nb_epoch': [500,1000],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_absolute_error',
                           cv =10)

#this might take a while 
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_










