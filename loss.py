import numpy as np

def mse_loss(X, y, theta):
    m = len(y)

    hypothesis = (theta[:][1] * X[:][1]) + theta[:][0]
    diff = np.sum((hypothesis - y) * (hypothesis - y))

    return 1/(2*m) * diff
