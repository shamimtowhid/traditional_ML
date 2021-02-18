import numpy as np


def mse_loss(X, y, theta):
    m = len(y)

    hypothesis = np.dot(X, theta)
    diff = np.sum(np.square(hypothesis-y.values.reshape(-1, 1)))

    return 1/(2*m) * diff
