# House Price Prediction with Linear Regression

This repository contains two examples of **Linear Regression** using scikit-learn in Python:

1. **Simple Linear Regression** → Predicting price based on **house size (m²)**.  
2. **Multiple Linear Regression** → Predicting price based on **house size (m²) + number of bedrooms**.  

The code is fully documented so you can understand the logic step by step.

---

## 1) Simple Linear Regression (1 feature)

```python
# Predict the price of a house based on its size (in square meters).

# This code is an example of simple linear regression (a single feature) 
# using the LinearRegression class from scikit-learn.
#
# One independent variable (X) → the house size in m².
# One dependent variable (y) → the house price.
# The model fits a straight line (linear function) that best describes the relationship between X and y.
# Then, it uses model.predict() to estimate the price of a new house (e.g., 90 m²).

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([[50], [60], [80], [100], [120]])
y = np.array([150000, 180000, 240000, 300000, 360000])

# Create the model (mathematical structure). It will look for the best fitting line (linear function).
# At this point, we are just creating an empty object of the LinearRegression class from scikit-learn.
# It represents the mathematical structure of linear regression, but it doesn’t have coefficients yet (β0, β1…).
model = LinearRegression()

# State before fitting: coefficients do not exist yet
state_before = getattr(model, "coef_", "Not trained yet")

# Train the model (fit coefficients and intercept)
# The model learns the relationship between X (independent variable) and y (target variable).
model.fit(X, y)

# State after fitting: intercept and coefficient found
intercept = model.intercept_
coefficient = model.coef_[0]

print("State BEFORE fitting:")
print(state_before)

print("\nState AFTER fitting:")
print("Intercept: " + str(intercept))
print("Coefficient: " + str(coefficient))

# Use the model to predict new values
# Now we can provide new values and get predictions.
# The model uses the coefficients learned during .fit().
# After .fit(), the model becomes a trained model, with parameters stored in:
#    model.coef_ (coefficients)
#    model.intercept_ (intercept)
X_new = np.array([[70], [90], [110]])
y_pred = model.predict(X_new)

print("\nPredictions for new values of X:")
for x_val, pred in zip(X_new.flatten(), y_pred):
    print(f"Predicted price for house of {x_val} m²: ${pred:.2f}")

# Workflow summary:
# -----------------
# model = LinearRegression() → just creates the structure.
# model.fit(X, y) → fits the coefficients β.
# model.predict(X_new) → applies the learned formula ŷ.    

# ---- PLOT ----
plt.figure(figsize=(8,6))

# Original data (scatter points)
plt.scatter(X, y, color="blue", label="Real data")

# Linear regression formula (for visualization on the plot)
# General form (prediction): ŷ = β0 + β1*x
formula = f"Price = {intercept:,.2f} + ({coefficient:,.2f} × Size_m²)"

# Generate points for the regression line (from min to max of X)
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)

# Regression line
plt.plot(x_range, y_range, color="red", linewidth=2, label="Regression line")

# Predicted points
plt.scatter(X_new, y_pred, color="green", marker="o", s=100, label="Predictions")

# Customization
plt.title("Simple Linear Regression - Price vs House Size")
plt.xlabel("House size (m²)")
plt.ylabel("House price ($)")
plt.legend()
plt.grid(True)

# Show the regression formula on the plot
plt.text(60, 200000, formula, fontsize=10, color="red")

plt.show()
