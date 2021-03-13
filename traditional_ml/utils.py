import numpy as np
import matplotlib.pyplot as plt

from traditional_ml.logistic_regression import sigmoid
from traditional_ml.linear_regression import gd, mse_loss


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
    '''
    This function initialize theta values to zeros for using in machine learning algorithm.

    Parameters:
        num_features (int): Number of features in the training data.

    Return:
        A numpy array of zeros. The shape depends on the number of features.
        For m number of feature, the return shape will be (m x 1)
    '''
    return np.zeros((num_features, 1))


def log_regression_predict(X, theta, num_cls=2):
    m, n = X.shape
    p = np.zeros((m, 1))

    h = sigmoid(X.dot(theta.T))

    if num_cls > 2:
        return h  # returns the probability of each class
    else:
        for it in range(0, h.shape[0]):
            if h[it] > 0.5:
                p[it, 0] = 1
            else:
                p[it, 0] = 0
        return p


def trainLinearRegression(Xtrain, y_train, lamda):
    feature = Xtrain.shape[1]
    theta = initialize_theta(feature)

    _, theta = gd(Xtrain, y_train, theta, 0.001, 200, lamda)

    return theta


def plot_learning_curve(X, y, Xval, yval, lamda):
    m = X.shape[0]
    y = y.values.reshape(-1, 1) if hasattr(y, 'values') else y.reshape(-1, 1)

    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))

    for idx, i in enumerate(range(1, m+1)):
        X_train = X[:i, :]
        y_train = y[:i]

        theta = trainLinearRegression(X_train, y_train, lamda)
        error_train[idx] = mse_loss(X_train, y_train, theta, 0)
        error_val[idx] = mse_loss(Xval, yval, theta, 0)

    plt.plot(error_train, label="train error")
    plt.plot(error_val, label="valid error")
    plt.title("Learning Curve")
    plt.legend()
    plt.xlabel('training data size')
    plt.ylabel('error')
