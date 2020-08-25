# simple linear regression
# predict salary based on year of experience


# importing liberary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1].values    # using values we get matrix from
y = dataset.iloc[:,1].values

# splitting the dataset into tranning dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 1/3,random_state = 0)  # test_sise  =0.3 = 1/3

"""
#  Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)  # fit scaler then transfrom 
X_test = sc_X.transform(X_test)  # here we don't need fit cause we already applied to X_train

# note == there is no need feature scaling in simple_LR model,THE LIBEARRY take care of feature scALING 
#           IN SIMPLE LINEAR REGRESSION 
"""

# Fitting Simple Linear Regression  to train set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)
 
# here LinearRegression fit in trannnig set,(which is actually straight line  )

# predicting the test set result
y_pred = regressor.predict(X_test)  # it"ll predict the values,


# visualize the Simple_linear regression(trainnig_set)
plt.scatter(X_train,y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train) , color = "green")
plt.title( 'Salary vs Expreience(Training_set)')
plt.xlabel('year of expreience')
plt.ylabel("salary")
plt.show()


# visualize test_set result

plt.scatter(X_test,y_test, color = "red")
plt.plot(X_test, regressor.predict(X_test) , color = "green")
plt.title( 'Salary vs Expreience(Training_set)')
plt.xlabel('year of expreience')
plt.ylabel("salary")
plt.show()


                                                                