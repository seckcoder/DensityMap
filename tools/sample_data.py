

import numpy as np


def gen_data(n):
    return np.vstack((
            np.random.multivariate_normal((0,0), [[1,0],[0,1]], n/2),
            np.random.multivariate_normal((10,10), [[1, 0],[0, 1]], n/2)))


def normalize(m):
    max_v = np.max(m)
    min_v = np.min(m)
    return (m - min_v) / (max_v - min_v)

for n in [10, 100, 1000]:
    m = gen_data(n)
    m1 = normalize(m[:,0])
    m2 = normalize(m[:,1])
    m = np.dstack((m1,m2))[0]
    np.savetxt("data/"+str(n) + ".txt",m,fmt='%10.5f', delimiter='\t')
