import pandas as pd

from plot_data import scatter_plot


if __name__=='__main__':
    #load data
    data = pd.read_csv('./ex1data1.txt', header= None)
    
    # visualize data
    scatter_plot(data[0], data[1], xlabel='population of city in 10,000s',
                    ylabel='profit in $10,000s', title='scatter plot of training data')
