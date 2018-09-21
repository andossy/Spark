#!/bin/python
# -*- coding: utf-8 -*-
import h5py
import time
import numpy as np
from collections import OrderedDict
from generate_geometry_file import geometry_file

def generate_rod_geometry_file(N_jsr, N_cleft, res=12, double=True,
                               basename="simpl_rod_geometry"):
    """
    Generates a h5 geometry file 
    """
    N_other = 1
    Nx, Ny = N_other, N_other
    voxels = np.zeros((Nx, Ny, 2 + N_jsr + N_cleft), dtype=np.uint8)
    
    # Voxel indices
    inds = OrderedDict()
    inds["nsr"] = 0
    inds["jsr"] = 1
    inds["cleft"] = 2
    inds["cyt"]  = 3

    # Assign voxels
    voxels[:,:,0] = inds["nsr"]
    voxels[:,:,1:N_jsr+1] = inds["jsr"]
    voxels[:,:,N_jsr+1:-1] = inds["cleft"]
    voxels[:,:,-1] = inds["cyt"]

    domain_list = inds.keys()
    domain_connections = [("nsr", "jsr"), ("cyt", "cleft")]
    connections = np.zeros((len(domain_list), len(domain_list)), dtype=np.uint8)
    for conn in domain_connections:
        assert len(conn) == 2
        assert conn[0] in domain_list
        assert conn[1] in domain_list
        ind0 = domain_list.index(conn[0])
        ind1 = domain_list.index(conn[1])
        connections[ind0, ind1] = 1
        connections[ind1, ind0] = 1

    # Double or single precision
    float_type = np.float64 if double else np.float32

    # Save data to file
    double_str = "_double" if double else ""

    # 1 RyR Channel
    RyRs = np.zeros((Nx*Ny, 6), dtype=np.uint16)
    bound_i = 0
    for i in range(Nx):
        for j in range(Ny):
            RyRs[bound_i,:] = (i, j, N_jsr, i, j, N_jsr+1)
            bound_i+=1

    # Save voxel and face data as a h5 file
    with geometry_file("{}{}.h5".format(basename, double_str)) as f:

        f.attrs.create("h", res, dtype=float_type)
        f.attrs.create("global_size", np.array([res, res, voxels.shape[0]*res],
                                               dtype=float_type))
        g = f.create_group("domains")
        g.attrs.create("num", len(inds), dtype=np.uint8) # cyt, jsr, tt
        for name, ind in inds.items():
            g.attrs.create("name_{}".format(ind), name)
        g.attrs.create("indices", np.array(inds.values(), dtype=np.uint8))
        g.create_dataset("voxels", data=voxels, compression="gzip",\
                                   dtype=np.uint8)

        # Add domain connections
        g.create_dataset("domain_connections", data=connections, dtype=np.uint8)

        faces = f.create_group("boundaries")
        faces.attrs.create("num", 1, dtype=np.uint8) # cyt, jsr, tt
        faces.attrs.create("name_0", "ryr")
        faces.create_dataset("ryr", data=RyRs, compression="gzip", \
                                    dtype=np.uint16)
        faces.attrs.create("type_0", "density"); # "discrete");
        
if __name__ == "__main__":

    generate_rod_geometry_file(10, 10, res=12, double=True,
                               basename="rod_geometry")
