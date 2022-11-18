import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def custom_accuracy(y_test, y_pred, thresold):
    right = 0

    l = len(y_pred)
    for i in range(0, l):
        if (abs(y_pred[i]-y_test[i]) <= thresold):
            right += 1
    return ((right/l)*100)


# Importing the dataset
dataset = pd.read_csv('t20.csv')
X = dataset.iloc[:, [7, 8, 9, 12, 13]].values
y = dataset.iloc[:, 14].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the dataset
lin = LinearRegression()
lin.fit(X_train, y_train)

# Testing the dataset on trained model
y_pred = lin.predict(X_test)
score = lin.score(X_test, y_test)*100
print("R square value:", score)
print("Custom accuracy:", custom_accuracy(y_test, y_pred, 20))

# Testing with a custom input
new_prediction = lin.predict(sc.transform(np.array([[100, 0, 13, 50, 50]])))
print("Prediction score:", new_prediction)
