import numpy as np
import scipy.optimize as op

from traditional_ml.utils import initialize_theta


def optimize_theta(X, y, loss_fn, theta, gradient_fn, lamda=0):
    result = op.minimize(fun=loss_fn,
                         x0=theta,
                         args=(X, y, lamda),
                         method='TNC',
                         jac=gradient_fn)
    return result.x, result.fun


def optimize_theta_for_multiclass(X, y, loss_fn, gradient_fn, cls_idx, lamda=0.1):
    num_feature = X.shape[1]
    num_class = len(cls_idx)
    theta = np.zeros((num_class, num_feature))  # +1 is for the inception in feature vector

    loss = []
    for i, cls in enumerate(cls_idx):
        lbl = np.array(y == cls).astype(int)
        init_theta = initialize_theta(num_feature)
        result = op.minimize(fun=loss_fn,
                             x0=init_theta,
                             args=(X, lbl, lamda),
                             method='TNC',
                             jac=gradient_fn)
        loss.append(result.fun)
        theta[i] = result.x

    return theta, loss
