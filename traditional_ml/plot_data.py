import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import contour, title, xlabel, ylabel, show

from traditional_ml.logistic_regression import map_feature


def scatter_plot(x, y, m='+', xlabel='X', ylabel='Y', title='scatter plot', show=True, legend=None):
    plt.scatter(x, y, marker=m)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if legend:
        plt.legend(legend)

    if show:
        plt.show()
    else:
        return plt


def plot(y, x=None, xlabel='X', ylabel='Y', title='plot', show=True, label='label'):
    if x is None:
        x = [i for i in range(len(y))]

    assert len(x) == len(y), 'for plotting, length must be equal'

    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if show:
        plt.show()
    else:
        return plt


def surface_plot(x, y, z):
    ax = plt.axes(projection='3d')

    ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_title('loss surface plot')
    plt.show()


def contour_plot(z, xlabel='X', ylabel='Y', title='contour plot',
                 levels=[10, 20, 30]):
    plt.contourf(z, levels=levels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()


def plot_decision_boundary(x, theta, xlabel='X', ylabel='Y', title='plot'):
    m, n = x.shape

    if n <= 3:
        plot_x = np.array([np.min(x[1])-2, np.max(x[1])+2])
        plot_y = (-1/theta[-1]) * (theta[1] * plot_x + theta[0])

        plot(plot_y, plot_x, xlabel, ylabel, title)

    else:
        print('Decision Boundary can be plot for only two features Dataset.')


def plot_non_linear_boundary(theta, degree=6, Xlabel='X', Ylabel='Y', plot_title='plot'):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))

    for i in range(len(u)):
        for j in range(len(v)):
            z[i, j] = (map_feature(np.array(u[i]), np.array(v[j]), degree).dot(np.array(theta)))

    z = z.T
    contour(u, v, z)
    title(plot_title)
    xlabel(Xlabel)
    ylabel(Ylabel)
    show()
