import numpy as np
from tqdm import tqdm

from traditional_ml.neural_network.loss import nn_loss


def optimize_theta(X, y, initial_nn_params, alpha, num_iters, Lambda, input_layer_size, hidden_layer_size, num_labels):
    Theta1 = initial_nn_params[:((input_layer_size+1) * hidden_layer_size)
                               ].reshape(hidden_layer_size, input_layer_size+1)
    Theta2 = initial_nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size+1)

    J_history = []

    for i in tqdm(range(num_iters)):
        nn_params = np.append(Theta1.flatten(), Theta2.flatten())
        cost, grad1, grad2 = nn_loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)[3:]
        Theta1 = Theta1 - (alpha * grad1)
        Theta2 = Theta2 - (alpha * grad2)
        J_history.append(cost)

    nn_paramsFinal = np.append(Theta1.flatten(), Theta2.flatten())
    return nn_paramsFinal, J_history
