
# Using RFR predict salary based on (position) dataset
# here dataset is nonlinear so we'll use kernel = ' rbf'



# importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

"""
# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)"""



# splitting data into train_test_split
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)"""


# fitting Random Forest to train dataset
from sklearn.tree import DecisionTreeRegressor
regressor =  DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y.ravel())


# predict the model
y_pred = regressor.predict(np.array([6.5]).reshape(-1,1))



# visualize the result
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X,y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'green')
plt.title('Truth or Bluff(Rabdom Forest Regression)')
plt.xlabel('Position-level')
plt.ylabel('Salary')
plt.show()

