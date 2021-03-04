import numpy as np


def rand_init_weight(Lin, Lout):
    """One effective strategy for choosing init is to base it on the number of units in the
       network. A good choice of epsilon is

       epsilon =√6/√(Lin+Lout)


       here Lin = sl and Lout = sl+1 are the number of units in the layers adjacent to Θ(l)
    """
    W = np.zeros((Lout, Lin+1))
    epsilon = np.sqrt(6)/np.sqrt(Lin+Lout)

    W = np.random.uniform(size=(Lout, Lin+1)) * 2 * epsilon - epsilon

    return W
