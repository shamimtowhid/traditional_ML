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
        #print('Loss: ', l_hist[-1])

    return l_hist, theta


def _gd(X, y, theta, alpha, num_iter):
    l_hist = []
    m = len(y)
    for _ in range(num_iter):
        hypothesis = (theta[:][1] * X[:][1]) + theta[:][0]
        diff = (hypothesis - y)

        gradient1 = np.sum(diff*X[:][0])
        gradient2 = np.sum(diff*X[:][1])

        theta[0][0] = theta[0][0] - alpha * ((1/m)*gradient1)
        theta[1][0] = theta[1][0] - alpha * ((1/m)*gradient2)

        l_hist.append(mse_loss(X, y, theta))
        #print('Loss: ', l_hist[-1])

    return l_hist, theta
