import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import numpy as np
import math
import argparse

def main(filename):
    x, y, z = np.loadtxt(filename, unpack = True)
    N = int(math.sqrt(len(z)))
    z = z.reshape(N, N)
    plt.imshow(z+10, extent = (np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
            cmap = cm.hot, norm = LogNorm())
    plt.colorbar()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="input file");
args = parser.parse_args();
if args.f:
    main(args.f)
