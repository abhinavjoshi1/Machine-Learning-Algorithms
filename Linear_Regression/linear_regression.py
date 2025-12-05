# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

kaggle_database = os.getenv("KAGGLE_DB")

training_db_path = os.path.join(kaggle_database + "/home-data-for-ml-course/train.csv")
training_dataset = pd.read_csv(training_db_path)

testing_db_path = os.path.join(kaggle_database + "/home-data-for-ml-course/test.csv")
testing_dataset = pd.read_csv(testing_db_path)

# Taking one feature only for prediction
x = training_dataset["GrLivArea"].values.reshape(-1, 1)
y = training_dataset["SalePrice"].values.reshape(-1, 1)

# Save mean & std from raw training data
x_mean = np.mean(x)
x_std = np.std(x)

# Normalize training data
x = (x - x_mean) / x_std

# number of samples
m = len(y)
W = np.random.rand(1, 1)   # weight
b = 0.0                    # bias

# Learning rate and epochs
alpha = 0.01
epochs = 1000
losses = []

for i in range(epochs):
    # Prediction
    y_pred = np.dot(x, W) + b
    
    # Error
    error = y_pred - y
    
    # Cost (MSE)
    cost = (1/(2*m)) * np.sum(error**2)
    losses.append(cost)
    
    # Gradients
    dW = (1/m) * np.dot(x.T, error)
    db = (1/m) * np.sum(error)
    
    # Update weights
    W -= alpha * dW
    b -= alpha * db
    
    if i % 100 == 0:
        print(f"Epoch {i}, Cost: {cost}")

# Plot training loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Training Loss")
plt.show()