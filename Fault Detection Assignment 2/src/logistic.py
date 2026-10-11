"""Task 3 - logistic regression written by hand (NumPy only)."""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def hypothesis(X, theta):
    return sigmoid(X @ theta)


def cost(X, y, theta):
    """J(theta) - binary cross-entropy. J(0) = log 2 = 0.693."""
    h = np.clip(hypothesis(X, theta), 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))


def fit_logistic(X, y, alpha, n_iters):
    """Batch gradient descent from theta = 0. history[0] = J(0); history[k] = J after k updates."""
    m = X.shape[0]
    theta = np.zeros(X.shape[1])
    history = [cost(X, y, theta)]
    for _ in range(n_iters):
        theta = theta + alpha * (X.T @ (y - hypothesis(X, theta))) / m
        history.append(cost(X, y, theta))
    return theta, np.array(history)


def predict(X, theta):
    return (hypothesis(X, theta) >= 0.5).astype(int)
