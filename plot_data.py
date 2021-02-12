import matplotlib.pyplot as plt

def scatter_plot(x, y, m='+', xlabel='X', ylabel='Y', title='scatter plot'):
    plt.scatter(x, y, marker=m)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()

def plot(y, x=None, xlabel='X', ylabel='Y', title='plot'):
    if not x:
        x = [i for i in range(len(y))]

    assert len(x) == len(y), 'for plotting, length must be equal'

    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
