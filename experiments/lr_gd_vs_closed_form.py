import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 1. Generate data
# -----------------------
np.random.seed(0)
n=100
X=np.linspace(-5000,5000,n)

true_w=3
true_b=2

noise=np.random.normal(0,2,n)
Y= true_w*X +true_b + noise


# -----------------------
# 2. Gradient Descent
# -----------------------
w=0.0
b=0.0

lr=0.01
epochs=1000

for _ in range(epochs):
    y_pred=w*X+b
    residual=Y-y_pred

    dw=-2*np.mean(X*residual)
    db=-2*np.mean(residual)

    w-=lr*dw
    b-=lr*db

w_gd=w
b_gd=b


# -----------------------
# 3. Closed Form
# -----------------------
X_design = np.column_stack([X, np.ones(len(X))])

theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y

w_cf = theta[0]
b_cf = theta[1]

X_design = np.column_stack([X, np.ones(len(X))])

theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ Y

w_cf = theta[0]
b_cf = theta[1]


# -----------------------
# 4. Compare Results
# -----------------------
print("Gradient Descent:")
print("w =", w_gd, "b =", b_gd)

print("\nClosed Form:")
print("w =", w_cf, "b =", b_cf)


# -----------------------
# 5. Plot
# -----------------------
plt.scatter(X, Y, label="Data")

plt.plot(X, w_gd * X + b_gd, label="Gradient Descent")
plt.plot(X, w_cf * X + b_cf, linestyle='dashed', label="Closed Form")

plt.legend()
plt.show()




'''increasing lr gD chaos
decreasing gD iterations, gD sucks
making X huge(-5000,5000), gD scaling issue

Optimization (GD) = Algebra (Normal Equation) = Geometry (Projection)'''