import numpy as np

from traditional_ml.logistic_regression.activation import sigmoid


def gradient(theta, X, y, lamda):
    X = X.values if hasattr(X, 'values') else X
    m, n = X.shape

    theta = theta.reshape((n, 1))
    thetaR = theta[1:, 0]

    y = y.values.reshape((m, 1)) if hasattr(y, 'values') else y.reshape((m, 1))

    h = sigmoid(X.dot(theta))
    delta = h - y
    sumdelta = delta.T.dot(X[:, 0])
    grad1 = (1.0 / m) * sumdelta

    XR = X[:, 1:X.shape[1]]
    sumdelta = delta.T.dot(XR)

    grad = (1.0 / m) * (sumdelta + lamda * thetaR)
    out_row = grad.values.shape[0] if hasattr(grad, 'values') else grad.shape[0]
    out_col = grad.values.shape[1] if hasattr(grad, 'values') else grad.shape[1]
    out = np.zeros((out_row, out_col + 1))

    out[:, 0] = grad1
    out[:, 1:] = grad

    return out.values.flatten() if hasattr(out, 'values') else out.flatten()


def logistic_loss(theta, X, y, lamda):
    m, n = X.shape

    theta = theta.reshape((n, 1))
    thetaR = theta[1:, 0]

    y = y.values.reshape((m, 1)) if hasattr(y, 'values') else y.reshape((m, 1))

    term1 = np.log(sigmoid(X.dot(theta)))
    term2 = np.log(1-sigmoid(X.dot(theta)))
    term3 = (lamda / (2.0 * m)) * (thetaR.T.dot(thetaR))

    term1 = term1.values.reshape((m, 1)) if hasattr(term1, 'values') else term1.reshape((m, 1))
    term2 = term2.values.reshape((m, 1)) if hasattr(term2, 'values') else term2.reshape((m, 1))

    term = y * term1 + (1 - y) * term2 - term3

    return -((np.sum(term))/m)
