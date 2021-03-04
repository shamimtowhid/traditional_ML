import numpy as np
from tqdm import tqdm

from traditional_ml.neural_network.loss import nn_loss


def optimize_theta(X, y, theta, lr, iterations, Lambda, input_layer_size, hidden_layer_size, num_labels):
    theta1 = theta[:((input_layer_size+1) * hidden_layer_size)
                   ].reshape(hidden_layer_size, input_layer_size+1)
    theta2 = theta[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size+1)

    loss_history = []

    for i in tqdm(range(iterations)):
        nn_params = np.append(theta1.flatten(), theta2.flatten())
        loss, grad1, grad2 = nn_loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[3:]
        theta1 = theta1 - (lr * grad1)
        theta2 = theta2 - (lr * grad2)
        loss_history.append(loss)

    final_nn_params = np.append(theta1.flatten(), theta2.flatten())
    return final_nn_params, loss_history
