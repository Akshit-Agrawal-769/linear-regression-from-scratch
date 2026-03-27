# import numpy as np
# import matplotlib.pyplot as plt

# x=np.linspace(-5,5,100)
# true_w=3
# true_b=2

# noise=np.random.normal(0,2,100)
# Y=true_w*x+true_b+noise

# w=0
# b=0
# lr=0.001
# lambda_ = 1

# loss=[]
# for i in range(1000):
#     residual=w*x+b-Y
#     dw=2*np.mean(x*residual)+2*lambda_*w
#     db=2*np.mean(residual)
#     w-=dw*lr
#     b-=db*lr
#     loss.append(np.mean(residual**2))

# print(w,b)
# plt.plot(loss)
# plt.title("Loss over iterations")
# plt.show()


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# True function (the "reality")
def true_function(x):
    return 0.5 * x**2 + 2*x + 1

x = np.linspace(-3, 3, 100)

# store predictions from many models
all_preds = []

# number of datasets
num_datasets = 50

for _ in range(num_datasets):
    noise = np.random.normal(0, 2, len(x))
    y = true_function(x) + noise

    # TRY DIFFERENT MODELS
    X = np.column_stack([x**i for i in range(1,10)]+ [np.ones(len(x))])

    lambda_=0.1
    I=np.eye(X.shape[1])
    theta = np.linalg.inv(X.T @ X + lambda_*I ) @ X.T @ y
    y_pred = X @ theta

    all_preds.append(y_pred)

all_preds = np.array(all_preds)

# Mean prediction (expected model)
mean_pred = np.mean(all_preds, axis=0)

# True curve
y_true = true_function(x)

# Bias^2
bias2 = np.mean((mean_pred - y_true)**2)

# Variance
variance = np.mean(np.var(all_preds, axis=0))

print("Bias^2:", bias2)
print("Variance:", variance)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(8,5))

# all models (faint)
for pred in all_preds:
    plt.plot(x, pred, color='gray', alpha=0.1)

# mean prediction
plt.plot(x, mean_pred, label="Mean Model", linewidth=2)

# true function
plt.plot(x, y_true, label="True Function", linewidth=2)


plt.legend()
plt.title("Bias-Variance Visualization")
plt.show()

# ===============================
# LAMBDA SWEEP EXPERIMENT
# ===============================

lambdas = np.logspace(-4, 3, 20)

bias_list = []
var_list = []

for lambda_ in lambdas:
    all_preds = []

    for _ in range(num_datasets):
        noise = np.random.normal(0, 2, len(x))
        y = true_function(x) + noise

        X = np.column_stack([x**i for i in range(1,10)] + [np.ones(len(x))])

        I = np.eye(X.shape[1])
        I[-1, -1] = 0   # don't regularize bias

        theta = np.linalg.inv(X.T @ X + lambda_ * I) @ X.T @ y
        y_pred = X @ theta

        all_preds.append(y_pred)

    all_preds = np.array(all_preds)

    mean_pred = np.mean(all_preds, axis=0)

    bias2 = np.mean((mean_pred - y_true)**2)
    variance = np.mean(np.var(all_preds, axis=0))

    bias_list.append(bias2)
    var_list.append(variance)

# -----------------------
# Plot Bias-Variance vs Lambda
# -----------------------
plt.figure(figsize=(8,5))

plt.plot(lambdas, bias_list, label="Bias^2")
plt.plot(lambdas, var_list, label="Variance")
plt.plot(lambdas, np.array(bias_list)+np.array(var_list), label="Total Error")

plt.xscale("log")
plt.xlabel("Lambda")
plt.ylabel("Error")
plt.title("Bias-Variance Tradeoff vs Lambda")
plt.legend()

plt.show()