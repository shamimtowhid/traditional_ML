import numpy as np

from traditional_ml.logistic_regression import sigmoid


def lin_regression_predict(X, theta):
    return np.dot(X, theta)


def feature_normalize(X):
    num_features = X.shape[1]

    mu = np.zeros((1, num_features))
    sigma = np.zeros((1, num_features))

    for i in range(num_features):
        single_feature = X[:][i]

        mu[0, i] = np.mean(single_feature)
        sigma[0, i] = np.std(single_feature)

    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma


def initialize_theta(num_features):
    return np.zeros((num_features, 1))

def log_regression_predict(X, theta):
    m, n = X.shape
    p = np.zeros((m, 1))

    h = sigmoid(X.dot(theta.T))

    for it in range(0, h.shape[0]):
        if h[it] > 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p
