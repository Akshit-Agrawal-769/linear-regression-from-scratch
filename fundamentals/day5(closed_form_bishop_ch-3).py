import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# -----------------------
# 1. Data (nonlinear truth)
# -----------------------
n = 100
x = np.linspace(-3, 3, n)

# true curve
y = 0.5 * x**2 + 2*x + 1 + np.random.normal(0, 1, n)

# -----------------------
# 2. Polynomial features
# -----------------------
X = np.column_stack([x, x**2, np.ones(len(x))])

# -----------------------
# 3. Closed form
# -----------------------
theta = np.linalg.inv(X.T @ X) @ X.T @ y

# -----------------------
# 4. Prediction
# -----------------------
y_hat = X @ theta

# -----------------------
# 5. Plot
# -----------------------
plt.scatter(x, y, label="Data")
plt.plot(x, y_hat, label="Fitted Curve")

plt.legend()
plt.show()
 
print("Theta:", theta)