import pandas as pd
import numpy as np

from plot_data import scatter_plot, plot
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
    plot(history, xlabel='iterations', ylabel='loss values', title='training loss')
