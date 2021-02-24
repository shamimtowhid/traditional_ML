import numpy as np

from traditional_ml.logistic_regression.activation import sigmoid


def gradient(X, y, theta):
    m, n = X.shape

    y = y.reshape((m,1))

    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = ((x.T).dot(sigmoid_x_theta-y))/m

    return grad.flatten()

def logistic_loss(X, y, theta):
    m, n = X.shape

    y = y.reshape((m,1))

    term1 = np.log(sigmoid(x.dot(theta)))
    term2 = np.log(1-sigmoid(x.dot(theta)))

    term1 = term1.reshape((m,1))
    term2 = term2.reshape((m,1))

    term = y * term1 + (1 - y) * term2

    return -((np.sum(term))/m)
