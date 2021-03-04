import numpy as np

from traditional_ml.logistic_regression import sigmoid


def sigmoid_grad(z):
    g = np.zeros(z.shape)

    simpleg = 1./(1. + np.exp(-1*z))
    diffg = 1-simpleg
    g = simpleg*diffg
    return g


def nn_loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    # Reshape nn_params back into the parameters Theta1 and Theta2
    Theta1 = nn_params[:((input_layer_size+1) * hidden_layer_size)].reshape(hidden_layer_size, input_layer_size+1)
    Theta2 = nn_params[((input_layer_size + 1) * hidden_layer_size):].reshape(num_labels, hidden_layer_size+1)

    m = X.shape[0]
    J = 0
    X = np.hstack((np.ones((m, 1)), X))
    y10 = np.zeros((m, num_labels))

    a1 = sigmoid(X @ Theta1.T)  # @ sign is used for matrix multiplication, introduced in python 3.5
    a1 = np.hstack((np.ones((m, 1)), a1))  # hidden layer
    a2 = sigmoid(a1 @ Theta2.T)  # output layer

    for i in range(1, num_labels+1):
        y10[:, i-1][:, np.newaxis] = np.where(y == i, 1, 0)
    for j in range(num_labels):
        J = J + sum(-y10[:, j] * np.log(a2[:, j]) - (1-y10[:, j])*np.log(1-a2[:, j]))

    cost = 1/m * J
    reg_J = cost + Lambda/(2*m) * (np.sum(Theta1[:, 1:]**2) + np.sum(Theta2[:, 1:]**2))

    # Implement the backpropagation algorithm to compute the gradients

    grad1 = np.zeros((Theta1.shape))
    grad2 = np.zeros((Theta2.shape))

    for i in range(m):
        xi = X[i, :]  # 1 X 401
        a1i = a1[i, :]  # 1 X 26
        a2i = a2[i, :]  # 1 X 10
        d2 = a2i - y10[i, :]
        d1 = Theta2.T @ d2.T * sigmoid_grad(np.hstack((1, xi @ Theta1.T)))
        grad1 = grad1 + d1[1:][:, np.newaxis] @ xi[:, np.newaxis].T
        grad2 = grad2 + d2.T[:, np.newaxis] @ a1i[:, np.newaxis].T

    grad1 = 1/m * grad1
    grad2 = 1/m*grad2

    grad1_reg = grad1 + (Lambda/m) * np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    grad2_reg = grad2 + (Lambda/m) * np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))

    return cost, grad1, grad2, reg_J, grad1_reg, grad2_reg
