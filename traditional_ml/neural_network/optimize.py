import scipy.optimize as op

from traditional_ml.neural_network.loss import nn_loss, nn_loss_grad


def optimize_theta(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    theta = (nn_params.T).ravel()
    result = op.minimize(fun=nn_loss, x0=theta, args=(nn_params,
                                     input_layer_size, hidden_layer_size, num_labels,
                                     X, y, lamda), method='CG', jac=nn_loss_grad
                    )

    return result.x, result.fun
