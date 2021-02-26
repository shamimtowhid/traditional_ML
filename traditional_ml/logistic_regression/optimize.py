import scipy.optimize as op


def optimize_theta(X, y, loss_fn, theta, gradient_fn, lamda=0):
    result = op.minimize(fun=loss_fn,
                         x0=theta,
                         args=(X, y, lamda),
                         method='TNC',
                         jac=gradient_fn)
    return result.x, result.fun
