import numpy as np

from traditional_ml.logistic_regression import sigmoid


def nn_predict(X: np.ndarray, theta_list: list) -> tuple:
    layer_out = X
    for i, theta in enumerate(theta_list):
        layer_out = sigmoid(np.insert(layer_out, 0, 1, axis=1).dot(theta.T))

    return np.argmax(layer_out), layer_out
