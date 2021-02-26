import numpy as np


def map_feature(x1, x2, degree=6):
    x1.shape = (x1.size, 1)
    x2.shape = (x2.size, 1)

    out = np.ones(shape=(x1[:, 0].size, 1))

    m, n = out.shape

    for i in range(1, degree + 1):
        for j in range(i + 1):
            r = (x1 ** (i - j)) * (x2 ** j)
            out = np.append(out, r, axis=1)

    return out
