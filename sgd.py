
import numpy as np
import matplotlib.pyplot as plt

dataset = np.loadtxt('assets/Income1.csv', delimiter=',', skiprows=1, usecols=[1, 2])

education, income = dataset.T

X = education.reshape(-1, 1)
y = income.reshape(-1, 1)

n_samples, n_features = X.shape

w = np.zeros(n_features)

epochs = 100
learning_rate = 0.001
batch_size = 5

mse_history = []

print('X.shape: ', X.shape)
print('n_samples is ', n_samples)
print('n_features is ', n_features)
print('w is ', w)
print('epochs is ', epochs)

for epoch in range(epochs):
    indices = np.arange(n_samples)

    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for index in range(0, n_samples, batch_size):
        X_batch = X_shuffled[index:index + batch_size]
        y_batch = y_shuffled[index:index + batch_size]

        estimated_y_batch = X_batch @ w
        error = estimated_y_batch - y_batch.flatten()
        gradient = (2 / len(X_batch)) * (X_batch.T @ error)

        w = w - learning_rate * gradient

    estimated_y = X @ w
    error = estimated_y - y.flatten()

    mse = np.mean(error ** 2)
    mse_history.append(mse)

    print(f'Epoch {epoch + 1}: w = {w[0]:.6f}, MSE = {mse:.6f}')

print('Final w:', w)
print('Final MSE:', mse_history[-1])

estimated_y = X @ w

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].scatter(education, income)
axes[0].plot(education, estimated_y, color='red')

axes[0].set_title("Linear Regression: Income x Education")
axes[0].set_ylabel("Income")
axes[0].set_xlabel("Education")

axes[1].plot(range(1, epochs + 1), mse_history, color='red')

axes[1].set_title("MSE During Training")
axes[1].set_ylabel("MSE")
axes[1].set_xlabel("Epoch")

plt.tight_layout()
plt.show()
