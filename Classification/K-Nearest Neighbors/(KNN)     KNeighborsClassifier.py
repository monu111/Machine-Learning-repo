# K-NN algorithm for classification

# importing liberary
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load dadaset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,4].values


# splitting dataset into train_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# fitting K-NN to training dataset
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 2)
classifier.fit(X_train,y_train)
# note = minkowski give euclidean distance  metrix with p = 2()

# predict the test dataset
y_pred = classifier.predict(X_test)


# creating confusion_matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# finding accuracy of test result
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# visualize the train_set
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train 

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(("red", "green")))
# set limit 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
# plotting scatter point
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j, 1],
                color =  ListedColormap(("red","green"))(i), label = j)
# plot
plt.title("KNeighborsClassifier(train_set)")
plt.xlabel("age")
plt.ylabel("estmated_salary")
plt.legend()
plt.show()


# visualize the test_set

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(("red","green")))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0],X_set[y_set == j ,1],
                color = ListedColormap(("red","green"))(i),label = j)

plt.title("KneighborsClassifier(test_set)")
plt.xlabel("Age")
plt.ylabel("estimated_salary")
plt.legend()
plt.show()




