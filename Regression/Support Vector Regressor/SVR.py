
# Using SVC predict salary based on (position) dataset
# here dataset is nonlinear so we'll use kernel = ' rbf'



# importing library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# we have to reshape single feature(1D array) to 2d array using this.
# otherwise we will  get error in feature scaling
y = y.reshape(-1,1)

# feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)



# splitting data into train_test_split
"""
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3, random_state = 0)"""


# fitting SVR in dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y.ravel())


# predict the model
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform((np.array([6.5]).reshape(1,-1)))))
# Note: Scale back the data to the original representation = inververse_transform 


# visualize the SVR
plt.scatter(X, y, c = 'red')
plt.plot(X, regressor.predict(X), c = 'green')
plt.title('truth or dare(SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



