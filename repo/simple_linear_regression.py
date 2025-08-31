from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Training data
X = np.array([[50], [60], [80], [100], [120]])
y = np.array([150000, 180000, 240000, 300000, 360000])

# Create the model
model = LinearRegression()

# State before fitting
state_before = getattr(model, "coef_", "Not trained yet")

# Train the model
model.fit(X, y)

# Coefficients after fitting
intercept = model.intercept_
coefficient = model.coef_[0]

print("State BEFORE fitting:")
print(state_before)

print("\nState AFTER fitting:")
print("Intercept: " + str(intercept))
print("Coefficient: " + str(coefficient))

# Predictions
X_new = np.array([[70], [90], [110]])
y_pred = model.predict(X_new)

print("\nPredictions for new values of X:")
for x_val, pred in zip(X_new.flatten(), y_pred):
    print(f"Predicted price for house of {x_val} m²: ${pred:.2f}")

# ---- PLOT ----
plt.figure(figsize=(8,6))
plt.scatter(X, y, color="blue", label="Real data")

formula = f"Price = {intercept:,.2f} + ({coefficient:,.2f} × Size_m²)"

x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range = model.predict(x_range)

plt.plot(x_range, y_range, color="red", linewidth=2, label="Regression line")
plt.scatter(X_new, y_pred, color="green", marker="o", s=100, label="Predictions")

plt.title("Simple Linear Regression - Price vs House Size")
plt.xlabel("House size (m²)")
plt.ylabel("House price ($)")
plt.legend()
plt.grid(True)
plt.text(60, 200000, formula, fontsize=10, color="red")

plt.show()
