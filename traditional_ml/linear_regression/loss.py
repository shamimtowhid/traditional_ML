import numpy as np


def mse_loss(X, y, theta, lamda=0):
    m = len(y)
    loss = 0

    y = y.values if hasattr(y, 'values') else y

    hypothesis = np.dot(X, theta)
    diff = np.sum(np.square(hypothesis-y.reshape(-1, 1)))

    theta_sum = np.sum(np.square(theta[:, 1:]))
    regularized_term = (lamda/(2*m)) * theta_sum

    loss = (1/(2*m) * diff) + regularized_term

    return loss


def mse_loss_gradient(X, y, theta, lamda=0):
    m = len(y)
    grad = np.zeros(theta.shape)

    regularized_term_grad = (lamda/m)*np.r_[[[0]], theta[1:]]

    hypothesis = np.dot(X, theta)
    grad = ((1/m) * X.T.dot(hypothesis-y)) + regularized_term_grad

    return grad
