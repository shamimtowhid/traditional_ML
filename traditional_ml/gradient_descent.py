import numpy as np

from loss import mse_loss


def gd(X, y, theta, alpha, num_iter):
    l_hist = []
    m = len(y)
    for _ in range(num_iter):
        hypothesis = np.dot(X, theta)
        diff = (hypothesis - y.values.reshape(-1, 1))

        gradients = np.sum(diff*X, axis=0).values.reshape(-1, 1)
        theta = theta - alpha * ((1/m)*gradients)

        l_hist.append(mse_loss(X, y, theta))
#        print('Loss: ', l_hist[-1])

    return l_hist, theta
