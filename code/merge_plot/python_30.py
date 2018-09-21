import sys
import numpy as np
import math
import time
import pylab
#from instant import inline_with_numpy
from random import random
import os
#import pdb

fig_size =  [20,10]
params = {'figure.figsize': fig_size}
pylab.rcParams.update(params)
h = 30;

nx = 66;
ny = 330;
nz = 330;

print nx,ny,nz
#one = np.ones((nx+2,ny+2,nz+2))
plane = np.ones((ny+2,nz+2))
Ca0 = 140e-3;
Ca_i_0 = Ca0*plane;
Ca_i_1 = Ca0*plane;
Ca_i_2 = Ca0*plane;
#Vfraction = (30./h)**3; # scaling of RyR when changing dx


T = 0
t = 0
counter = 0;
mean = [];
DT = 0.01; # plotting time step

use_instant = 0
num_threads = 0
t00 = time.time()
t_ryr = 0.0
t_conc = 0.0
#while t<T:
pylab.figure()
for TT in range(4):
	Ca_i_0 = np.loadtxt("merge_dir_%d_%d_%d_%d/Ca0_T%d_merge.np"%(h,nx,ny,nz,TT))
	#Ca_i_1 = np.loadtxt("merge_dir_%d_%d_%d_%d/Ca0_T%d_rank%d_merge.np"%(h,nx,ny,nz,TT,rank1))
	#Ca_i_2 = np.loadtxt("merge_dir_%d_%d_%d_%d/Ca0_T%d_rank%d_merge.np"%(h,nx,ny,nz,TT,rank2))
	print "frame",TT 
	#pylab.interactive(True)
	pylab.clf()
	pylab.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
	pylab.subplot(121)
	pylab.imshow(Ca_i_0[1:-1,1:-1], extent=[0,10,0,10])
	pylab.clim([0.1,20])
	pylab.colorbar()
	pylab.xlabel("[um]")
	pylab.ylabel("[um]")
	pylab.text(0.0, -0.10, 'Max: %.1f um' % Ca_i_0.max(), horizontalalignment='left', transform = pylab.gca().transAxes)

	pylab.subplot(122)
	pylab.imshow(Ca_i_0[1:-1,1:-1], extent=[0,10,0,10])
        pylab.clim([0.05,1])
	pylab.colorbar()
	pylab.xlabel("[um]")
	pylab.ylabel("[um]")
        if TT<=3:
            outfigname="R%dnm_%dFrame.png"%(h,TT)
            pylab.savefig(outfigname)
	#pylab.draw() 

pylab.show()       
#gets()
