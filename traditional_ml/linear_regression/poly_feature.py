import numpy as np


def poly_features(X, p):
    X_poly = X

    for i in range(2, p+1):
        X_poly = np.append(X_poly, X**i, axis=1)

    return X_poly
