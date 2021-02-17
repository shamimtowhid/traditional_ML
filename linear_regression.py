import pandas as pd
import numpy as np

from plot_data import scatter_plot, plot, surface_plot, contour_plot
from loss import mse_loss
from gradient_descent import gd
from utils import regression_predict, feature_normalize, initialize_theta_regression

if __name__=='__main__':
    #load data
    data = pd.read_csv('./ex1data2.txt', header= None)

    # visualize data
    #scatter_plot(data[0], data[1], xlabel='population of city in 10,000s',
    #                ylabel='profit in $10,000s', title='scatter plot of training data')

    m = len(data[0])
    #X = pd.DataFrame()
    #X[0] = np.ones((m))
    #X[1] = data[:][0]
    #y = data[:][1]
    #num_feature = X.shape[1]
    X_norm, mu, sigma = feature_normalize(data.loc[:, 0:1])
    X = pd.DataFrame()

    X[0] = np.ones((m))
    X[1] = X_norm[0]
    X[2] = X_norm[1]
    num_feature = X.shape[1]

    y = data[data.columns[-1]]

    theta = initialize_theta_regression(num_feature)
    iterations = 500
    alpha = 0.01

    history, theta = gd(X, y, theta, alpha, iterations)
    plot(history, xlabel='iterations', ylabel='loss values', title='training loss')

#    print("For population =35,000, we predict a profit of {}".format(
#            regression_predict(np.array([1, 3.5]), theta)*10000))

#    print("For population = 70,000, we predict a profit of {}".format(
#            regression_predict(np.array([1, 7.0]), theta)*10000))

#    X_norm, mu, sigma = feature_normalize(X)

#    print(X_norm.head())
#    print(mu, sigma)
    # visualize training data with linear regression fit
#    plt = scatter_plot(data[0], data[1], show=False)
#    plot(X[1], regression_predict(X, theta), xlabel='population of city in 10,000s',
#                    ylabel='profit in $10,000s', title='Training Data with regression fit')

    # visualize loss surface
#    theta0 = np.linspace(-10, 10, 100)
#    theta1 = np.linspace(-1, 4, 100)

#    losses = np.zeros((len(theta0), len(theta1)))

#    for i in range(len(theta0)):
#        for j in range(len(theta1)):
#            t = np.array([theta0[i], theta1[j]]).reshape(2,1)
#            losses[i][j] = mse_loss(X, y, t)

    #surface_plot(theta0, theta1, losses)
   
    # visualize contour plot
    #contour_plot(losses, levels=np.logspace(-2, 3, 20))
