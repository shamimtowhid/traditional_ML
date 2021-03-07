from tqdm import tqdm

from traditional_ml.linear_regression.loss import mse_loss, mse_loss_gradient


def gd(X, y, theta, alpha, num_iter, lamda=0):
    l_hist = []
    y = y.values.reshape(-1, 1) if hasattr(y, 'values') else y.reshape(-1, 1)

    for _ in tqdm(range(num_iter)):
        l_hist.append(mse_loss(X, y, theta, lamda))
        gradients = mse_loss_gradient(X, y, theta, lamda)
        theta = theta - (alpha * gradients)

    return l_hist, theta
