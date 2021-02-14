import pandas as pd
import numpy as np

from plot_data import scatter_plot, plot, surface_plot, contour_plot
from loss import mse_loss
from gradient_descent import gd

if __name__=='__main__':
    #load data
    data = pd.read_csv('./ex1data1.txt', header= None)

    # visualize data
    #scatter_plot(data[0], data[1], xlabel='population of city in 10,000s',
    #                ylabel='profit in $10,000s', title='scatter plot of training data')

    m = len(data[0])
    X = pd.DataFrame()
    X[0] = np.ones((m))
    X[1] = data[:][0]
    y = data[:][1]
    theta = np.zeros((2, 1))

    iterations = 1500
    alpha = 0.01

    history, theta = gd(X, y, theta, alpha, iterations)
    #plot(history, xlabel='iterations', ylabel='loss values', title='training loss')

    # visualize training data with linear regression fit
    #plt = scatter_plot(data[0], data[1], show=False)
    #plot(X[1],np.dot(X, theta), xlabel='population of city in 10,000s',
    #                ylabel='profit in $10,000s', title='Training Data with regression fit')

    # visualize loss surface
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)

    losses = np.zeros((len(theta0), len(theta1)))

    for i in range(len(theta0)):
        for j in range(len(theta1)):
            t = np.array([theta0[i], theta1[j]]).reshape(2,1)
            losses[i][j] = mse_loss(X, y, t)

    #surface_plot(theta0, theta1, losses)
   
    # visualize contour plot
    contour_plot(losses, levels=np.logspace(-2, 3, 20))
