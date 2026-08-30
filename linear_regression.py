import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('assets/Income1.csv', delimiter=',', skiprows=1, usecols=[1, 2])

education, income = dataset.T

x = education.reshape(-1, 1)
y = income.reshape(-1, 1)

xt_x = x.T @ x
xt_y = x.T @ y

w = np.linalg.inv(xt_x) @ xt_y

estimated_y = w * x

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

fig.canvas.manager.set_window_title("Linear Regression")

axes[0].scatter(education, income)
axes[0].plot(education, estimated_y, color='red')

axes[0].set_title("Linear Regression: Income x Education")
axes[0].set_ylabel("Income")
axes[0].set_xlabel("Education")

w_values = np.linspace(0, 6, 100)

mse_values = []

for w_value in w_values:
    error_estimated_y = w_value * x
    error = error_estimated_y - y
    square_error = error ** 2
    square_error_sum = np.sum(square_error)
    mse_value = square_error_sum / len(y)

    mse_values.append(mse_value)

min_w_value_index = np.argmin(mse_values)
min_w_value = w_values[min_w_value_index]

min_mse_value = mse_values[min_w_value_index]

regression_error = estimated_y - y
regression_square_error = regression_error ** 2
regression_square_error_sum = np.sum(regression_square_error)
regression_mse_value = regression_square_error_sum / len(y)

axes[1].scatter(w, regression_mse_value)
axes[1].plot(w_values, mse_values, color="red")

axes[1].set_title("MSE as a Function of w")
axes[1].set_ylabel("MSE")
axes[1].set_xlabel("w")


plt.tight_layout()

plt.show()
