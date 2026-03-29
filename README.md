# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. **Initialize Parameters**
   Initialize the model parameters (weights and bias) with small random values and choose a learning rate (η).

2. **Input Training Data**
   Provide the dataset with input features (X) and target values (y), and optionally split into training and testing sets.

3. **Iterate Over Data (Epochs)**
   For a fixed number of epochs, loop through each training sample one by one.

4. **Update Weights Using SGD Rule**
   For each sample, compute the prediction error and update the weights and bias using:

   * Gradient of loss function (usually Mean Squared Error)
   * Update rule:
     ( w = w - \eta \cdot \nabla L )

5. **Evaluate the Model**
   After training, predict outputs on test data and evaluate performance using metrics like:

   * Mean Squared Error (MSE)
   * R² Score

## Program:

Developed by: PARUVATHA VARSHINI P S
RegisterNumber:  212225100033
```# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv("CarPrice_Assignment (1).csv")
print(data.head())
print(data.info())

# Data preprocessing
# Dropping unnecessary columns and handling categorical variables
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)

# Splitting the data into features and target variable
X = data.drop('price', axis=1)
y = data['price']

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creating the SGD Regressor model
sgd_model = SGDRegressor(max_iter=1000,tol=1e-3)

#Fitting the model on the training data
sgd_model.fit(X_train, y_train)

#Making predictions
y_pred = sgd_model.predict(X_test)

#Evaluating model performance
mse = mean_squared_error(y_test, y_pred)
print('MSE= ',mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print('MAE= ',mean_absolute_error(y_test, y_pred))
print(f"R2: {r2_score(y_test, y_pred):.4f}")

# Print evaluation metrics
print('Name: ')
print('Reg. No: ')
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

# Visualizing actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices using SGD Regressor")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Perfect prediction line
plt.show()
```

## Output:
<img width="989" height="622" alt="Screenshot 2026-02-12 161428" src="https://github.com/user-attachments/assets/59630de8-58d6-4829-93d4-4dd9891977d6" />
<img width="608" height="718" alt="image" src="https://github.com/user-attachments/assets/8b7f8931-73fa-427d-88ce-bf717747d6cc" />
<img width="965" height="385" alt="image" src="https://github.com/user-attachments/assets/c3503e6e-0aa6-4978-8e39-bb2801590ec9" />
<img width="910" height="565" alt="image" src="https://github.com/user-attachments/assets/a0864071-31b1-4638-a2c9-dc40ea0c9c5c" />

## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
