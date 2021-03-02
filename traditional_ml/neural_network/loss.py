import numpy as np

from traditional_ml.logistic_regression import sigmoid


def sigmoid_grad(z):
    g = np.zeros(z.shape)

    simpleg = 1./(1. + np.exp(-1*z))
    diffg = 1-simpleg
    g = simpleg*diffg
    return g


def nn_loss_grad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    theta1 = nn_params[0]
    theta2 = nn_params[1]

    m, n = X.shape

    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)

    X = np.insert(X, 0, 1, axis=1)
    Z2 = X.dot(theta1.T)
    layer2 = sigmoid(Z2)
    layer2 = np.insert(layer2, 0, 1, axis=1)
    hypothesis = sigmoid(layer2.dot(theta2.T))

    temp = np.zeros((m, num_labels))
    for i in range(m):
        # -1 in the following line is for matching the index with octave indexing
        temp[i, y[i]-1] = 1
    y = temp

    sigma3 = hypothesis - y
    sigma2 = (sigma3.dot(theta2) * sigmoid_grad(np.insert(Z2, 0, 1, axis=1)))[:, 1:]

    delta1 = (sigma2.T).dot(X)
    delta2 = (sigma3.T).dot(layer2)

    tmp_theta1 = np.insert(theta1[:, 1:], 0, 0, axis=1)
    tmp_theta2 = np.insert(theta2[:, 1:], 0, 0, axis=1)
    theta1_grad = delta1/m + (lamda/m) * tmp_theta1
    theta2_grad = delta2/m + (lamda/m) * tmp_theta2

    grad = np.array([theta1_grad[:], theta2_grad[:]])

    return grad.flatten()


def nn_loss(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lamda):
    theta1 = nn_params[0]
    theta2 = nn_params[1]

    m, n = X.shape


    X = np.insert(X, 0, 1, axis=1)
    Z2 = X.dot(theta1.T)
    layer2 = sigmoid(Z2)
    layer2 = np.insert(layer2, 0, 1, axis=1)
    hypothesis = sigmoid(layer2.dot(theta2.T))
# create one hot vector
    temp = np.zeros((m, num_labels))
    for i in range(m):
        # -1 in the following line is for matching the index with octave indexing
        temp[i, y[i]-1] = 1
    y = temp

    fsum = np.sum((temp*np.log(hypothesis)) + ((1-temp)*np.log(1-hypothesis)))
    ssum = np.sum(fsum)

    theta1[:, 0] = 0
    theta2[:, 0] = 0

    ksum = np.sum(np.square(theta1))
    jsum = np.sum(ksum)
    ksum2 = np.sum(np.square(theta2))
    jsum2 = np.sum(ksum2)

    regularized_term = lamda/(2.*m)*(jsum+jsum2)

    loss = (-(1./m)*ssum)+regularized_term

    return loss
