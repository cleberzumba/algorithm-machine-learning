import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Data (X has 2 features: size and bedrooms)
X = np.array([[50, 1], [60, 2], [80, 2], [100, 3], [120, 3], [150, 4],])
y = np.array([150_000, 185_000, 250_000, 320_000, 380_000, 480_000])

# Train the model
model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coef_size, coef_bedrooms = model.coef_

print("Intercept (β0):", intercept)
print("Coefficients (β1, β2):", coef_size, coef_bedrooms)
print(f"Equation of the plane: price = {intercept:,.2f} + ({coef_size:,.2f} × size_m²) + ({coef_bedrooms:,.2f} × bedrooms)")

# Create a grid (mesh) for visualization
size_range = np.linspace(X[:,0].min()-5, X[:,0].max()+5, 40)
bed_range  = np.linspace(X[:,1].min()-0.5, X[:,1].max()+0.5, 40)
S, B = np.meshgrid(size_range, bed_range)
grid = np.c_[S.ravel(), B.ravel()]
Z = model.predict(grid).reshape(S.shape)

# 3D Plot: real points + regression plane
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:,0], X[:,1], y, s=50, label="Real data")
ax.plot_surface(S, B, Z, alpha=0.4)

# Predict new points
X_new = np.array([[70, 2], [90, 3], [110, 3]])
y_new = model.predict(X_new)
ax.scatter(X_new[:,0], X_new[:,1], y_new, marker="o", s=100, label="Predictions")

for (sx, qb), py in zip(X_new, y_new):
    ax.text(sx, qb, py, f"$ {py:,.0f}", fontsize=9)

# Plot customization
ax.set_xlabel('Size (m²)')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price ($)')
ax.set_title('Multiple Linear Regression: Price ~ Size + Bedrooms')
ax.view_init(elev=20, azim=35)
ax.legend()

plt.tight_layout()
plt.show()
