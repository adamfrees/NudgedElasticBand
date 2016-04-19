import numpy as np
import matplotlib.pyplot as plt
from NEB import *

def potential(x,y):
    return 0.5*np.cos(x/2.) - np.cos(y/4.)+y*5.e-2

def plotResults(potFunc,points):
    dx, dy = 0.2, 0.2
    y, x = np.mgrid[slice(-10, 10 + dy, dy),
                    slice(-10, 10 + dx, dx)]
    z = np.vectorize(potFunc)(x,y)
    z = z[:-1, :-1]
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    plt.pcolor(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
    plt.plot(*zip(*points),linewidth=2.,marker='o',markersize=15.)
    plt.show()

#plotPotential()
results = NEB(51,np.array([-2.*np.pi,0]),np.array([2.*np.pi,0]),potential)
toPlot = map(lambda x: x.position,results)
plotResults(potential,toPlot)
