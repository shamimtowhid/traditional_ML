import matplotlib.pyplot as plt

def scatter_plot(x, y, m='+', xlabel='X', ylabel='Y', title='scatter plot'):
    plt.scatter(x, y, marker=m)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.show()
