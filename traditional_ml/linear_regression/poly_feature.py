import numpy as np


def poly_features(X, p=8):
    X_poly = X.reshape(-1, 1)
    X_vec = X.reshape(-1, 1)

    for i in range(2, p+1):
        X_poly = np.append(X_poly, X_vec**i, axis=1)

    return X_poly
