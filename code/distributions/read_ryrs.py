import numpy as np
import matplotlib.pyplot as plt
cmap = plt.get_cmap("binary")

# Read indices for sarcomere 0
basename = "indices_sarc_%d_"
indi = np.fromfile(open("i_RyR.dat"), sep="\n", dtype=np.int32)
indj = np.fromfile(open("j_RyR.dat"), sep="\n", dtype=np.int32)

# 2D representation of the RyRs
# Each voxel containing a RyR is 30x30nm meaning that the
# whole sarcomere is ~ 10x10um
Nxy = 334

RyR0 = np.zeros((Nxy, Nxy))
RyR0[indi, indj] = 1

# Read indices for sarcomere 1
indi = np.fromfile(open("i_RyR.dat"), sep="\n", dtype=np.int32)
indj = np.fromfile(open("j_RyR.dat"), sep="\n", dtype=np.int32)

RyR1 = np.zeros((Nxy, Nxy))
RyR1[indi, indj] = 1

# Visualize the two index sets
plt.subplot(121)
plt.imshow(RyR0, interpolation="nearest", origin="lower",\
             extent=[0,Nxy*0.030,0,Nxy*0.030], cmap=cmap)
plt.subplot(122)
plt.imshow(RyR1, interpolation="nearest", origin="lower",\
             extent=[0,Nxy*0.030,0,Nxy*0.030], cmap=cmap)

plt.show()

# Combine into volumetric distribution
# Each voxel is again 30 nm wide. Each sarcomere is 2um long, and the CRU
# are distributed along the Z-line which is positioned in the center of
# each sarcomere (at z=1um and z=3um)

Nz = 4000/30 + 1
volume = np.zeros((Nz, Nxy, Nxy))
indk0 = 1000/30 + 1
indk1 = 3000/30 + 1

volume[indk0,:,:] = RyR0
volume[indk1,:,:] = RyR1

# TODO: Vizualize 3D data with some volumetric renderer
