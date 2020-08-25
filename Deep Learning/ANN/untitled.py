# Artificial Neural Network

# importing the libraries
import numpy as np
import tensorflow as tf
import pandas as pd


# importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,13].values

# Labelencoding the dataset
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])

# OneHotEncoding for 'geography'
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('geography', OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Part 2  building ANN 
# intialize the ANN

ann = tf.keras.Sequential()

# adding input layer and 1st hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

# adding second  Hidden layer
ann.add(tf.keras.layers.Dense(units = 6 , activation = 'relu'))

# adding output layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'sigmoid'))

#  Part 3 train Ann for trainning set

# intailze the compiler
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# train dataset with ANN
ann.fit(X_train, y_train)
















