import numpy as np

from traditional_ml.logistic_regression.activation import sigmoid


def gradient(theta, X, y):
    m, n = X.shape

    theta = theta.reshape((n, 1))
    y = y.values.reshape((m, 1))

    sigmoid_x_theta = sigmoid(X.dot(theta))
    grad = ((X.T).dot(sigmoid_x_theta-y))/m

    return grad.values.flatten()


def logistic_loss(theta, X, y):
    m, n = X.shape

    theta = theta.reshape((n, 1))
    y = y.values.reshape((m, 1))

    term1 = np.log(sigmoid(X.dot(theta)))
    term2 = np.log(1-sigmoid(X.dot(theta)))

    term1 = term1.values.reshape((m, 1))
    term2 = term2.values.reshape((m, 1))

    term = y * term1 + (1 - y) * term2

    return -((np.sum(term))/m)
