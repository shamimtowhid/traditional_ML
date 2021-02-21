import numpy as np


def regression_predict(X, theta):
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


def initialize_theta_regression(num_features):
    return np.zeros((num_features, 1))
