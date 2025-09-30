import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

train_data = {
    "GrLivArea": [10, 15, 20, 25, 30],
    "GarageArea": [2, 3, 4, 5, 6],
    "SalePrice": [150, 180, 200, 240, 280]
}

test_data = {
    "GrLivArea": [12, 16, 21, 26, 31],
    "GarageArea": [2, 3, 4, 5, 6],
    "SalePrice": [155, 185, 205, 245, 285]
}

kaggle_db = os.getenv("KAGGLE_DB")

training_db_path = os.path.join(kaggle_db + "/home-data-for-ml-course/train.csv")
training_ds = pd.read_csv(training_db_path)
# training_ds = pd.DataFrame(train_data)

testing_db_path = os.path.join(kaggle_db + "/home-data-for-ml-course/test.csv")
testing_ds = pd.read_csv(testing_db_path)
# testing_ds = pd.DataFrame(test_data)

X = training_ds[["GrLivArea", "GarageArea"]].values
# X = training_ds[["GrLivArea"]].values   # shape (m, 1)
Y = training_ds[["SalePrice"]].values     # shape (m,1)
# print(X)
# print(type(X))
# print(X.shape)
# sys.exit()

# Hypothesis space -> All straight lines
# h(x) = theta + theta1.x1 + theta2.x2 ... thetan.xn

m = len(Y)
X = np.c_[np.ones((m, 1)), X]

# initial values
theta = np.zeros((X.shape[1], 1))        # vector of theta

alpha = 0.000000001


def simple_linear_regression(X, Y, theta, alpha, epochs):
    
    m = len(Y)
    for _ in range(epochs):
        y_hat = X.dot(theta)
        
        residual = y_hat - Y
        
        
        # derivative of cost function
        dj = (2/m)*X.T.dot(residual)
        
        # update
        theta = theta - alpha*dj
    
    return theta
     
epochs = 1000
theta = simple_linear_regression(X, Y, theta, alpha, epochs)
print("Learned parameters (theta):")
print(theta)




