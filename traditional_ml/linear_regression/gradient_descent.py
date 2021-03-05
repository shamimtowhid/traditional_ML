import numpy as np
from tqdm import tqdm

from traditional_ml.linear_regression.loss import mse_loss, mse_loss_gradient


#def gd(X, y, theta, alpha, num_iter):
#    l_hist = []
#    m = len(y)
#    for _ in range(num_iter):
#        hypothesis = np.dot(X, theta)
#        diff = (hypothesis - y.values.reshape(-1, 1))
#
#        gradients = np.sum(diff*X, axis=0).values.reshape(-1, 1)
#        theta = theta - alpha * ((1/m)*gradients)
#
#        l_hist.append(mse_loss(X, y, theta))
#        print('Loss: ', l_hist[-1])
#
#    return l_hist, theta


def gd(X, y, theta, alpha, num_iter, lamda=0):
    l_hist = []
    m = len(y)
    y = y.values if hasattr(y, 'values') else y

    for _ in tqdm(range(num_iter)):
        l_hist.append(mse_loss(X, y, theta, lamda))

        gradients = mse_loss_gradient(X, y, theta, lamda)
        theta = theta - (alpha * gradients)

    return l_hist, theta
