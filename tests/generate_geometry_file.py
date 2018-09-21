#!/bin/python
# -*- coding: utf-8 -*-
import h5py
import time
import numpy as np
from collections import OrderedDict

class geometry_file:
    def __init__(self, filename):
        self.old_file = None
        self.filename = filename
    
    def __enter__(self):
        import os
        import shutil
        # If filename already exist
        if os.path.isfile(self.filename):
            self.old_file = "_old_"+self.filename
            shutil.move(self.filename, self.old_file)
        
        # Open a h5 file
        self.f = h5py.File(self.filename)
        return self.f 

    def __exit__(self, type, value, traceback):
        import os
        import shutil
        self.f.close()
        if type is None:
            if self.old_file:
                os.unlink(self.old_file)
        else:
            shutil.move(self.old_file, self.filename)

def generate_cru_geometry_file(RyR_data, Nz=28, RyR_size=36, h_geom=12, cleft_hight=12,
                               double=True, basename="sarcomere_geometry", save2d_pdf=False):
    """
    Generates a h5 geometry file from a 2D array of RyR positions
    """

    t0 = time.time()
    if not (isinstance(RyR_data, np.ndarray) and len(RyR_data.shape) ==2 and
            RyR_data.dtype==np.bool):
        raise TypeError("Expected a 2D numpy bool array as first argument")

    assert RyR_size % h_geom == 0

    # Geometry
    h_geom = 12
    cleft_size = cleft_hight/h_geom
    subdivide = RyR_size / h_geom
    Nx, Ny = RyR_data.shape

    x_domain_size = Nx*RyR_size
    y_domain_size = Ny*RyR_size
    z_domain_size = Ny*RyR_size
    
    indxm = np.arange(0,Nx-1)
    indxp = np.arange(1,Nx)
    indym = np.arange(0,Ny-1)
    indyp = np.arange(1,Ny)

    # Voxel indices
    inds = OrderedDict()
    inds["cyt"] = 0
    inds["cleft"] = 1
    inds["jsr"] = 2
    inds["tt"]  = 3
    #inds["nsr"] = 4

    domain_list = inds.keys()
    domain_connections = [("cyt", "cleft")]
    connections = np.zeros((len(domain_list), len(domain_list)), dtype=np.uint8)
    for conn in domain_connections:
        assert len(conn) == 2
        assert conn[0] in domain_list
        assert conn[1] in domain_list
        ind0 = domain_list.index(conn[0])
        ind1 = domain_list.index(conn[1])
        connections[ind0, ind1] = 1
        connections[ind1, ind0] = 1

    # Create the jSR 2D array
    # 1 Neighborhood around each RyR
    # FIXME: Should we increase this?
    jSR_data = RyR_data.copy()
    jSR_data[indxm, :] += RyR_data[indxp, :]
    jSR_data[indxp, :] += RyR_data[indxm, :]
    jSR_data[:, indym] += RyR_data[:, indyp]
    jSR_data[:, indyp] += RyR_data[:, indym]

    jSR_data2D = np.zeros((Nx*subdivide, Ny*subdivide), dtype=np.bool)
    RyR_data2D = np.zeros((Nx*subdivide, Ny*subdivide), dtype=np.bool)

    RyR_data2D[1::subdivide, 1::subdivide] = RyR_data
    for i in range(subdivide):
        for j in range(subdivide):
            jSR_data2D[i::subdivide, j::subdivide] = jSR_data

    # Smoothen the corners a bit

    # Down left corners
    down_diag = jSR_data[1:, :-1]*jSR_data[:-1, 1:]
    jSR_data2D[2:-subdivide:subdivide, 2:-subdivide:subdivide] += down_diag
    jSR_data2D[1:-subdivide:subdivide, 2:-subdivide:subdivide] += down_diag
    jSR_data2D[2:-subdivide:subdivide, 1:-subdivide:subdivide] += down_diag

    # Top right corners
    jSR_data2D[subdivide::subdivide, subdivide::subdivide]   += down_diag
    jSR_data2D[subdivide+1::subdivide, subdivide::subdivide] += down_diag
    jSR_data2D[subdivide::subdivide, subdivide+1::subdivide] += down_diag

    # Down right corners
    diag = jSR_data[:-1, :-1]*jSR_data[1:, 1:]
    jSR_data2D[2:-subdivide:subdivide, subdivide::subdivide]   += diag
    jSR_data2D[1:-subdivide:subdivide, subdivide::subdivide]   += diag
    jSR_data2D[2:-subdivide:subdivide, subdivide+1::subdivide] += diag

    # Top left corners
    jSR_data2D[subdivide::subdivide, 2:-subdivide:subdivide]   += diag
    jSR_data2D[subdivide+1::subdivide, 2:-subdivide:subdivide] += diag
    jSR_data2D[subdivide::subdivide, 1:-subdivide:subdivide]   += diag

    # Remove edges
    jSR_data2D[0,:]  = 0
    jSR_data2D[-1,:] = 0
    jSR_data2D[:,0]  = 0
    jSR_data2D[:,-1] = 0

    # Schrink jSR
    jSR_data2D_copy = jSR_data2D.copy()
    jSR_data2D[:,:-1] *= jSR_data2D_copy[:,1:]
    jSR_data2D[:,1:] *=  jSR_data2D_copy[:,:-1]
    jSR_data2D[:-1,:] *= jSR_data2D_copy[1:,:]
    jSR_data2D[1:,:] *=  jSR_data2D_copy[:-1,:]

    # Create a even more schrinked top of jSR
    jSR_data2D_top = jSR_data2D.copy()
    jSR_data2D_top[:,:-1] *= jSR_data2D[:,1:]
    jSR_data2D_top[:,1:] *=  jSR_data2D[:,:-1]
    jSR_data2D_top[:-1,:] *= jSR_data2D[1:,:]
    jSR_data2D_top[1:,:] *=  jSR_data2D[:-1,:]

    if save2d_pdf:
        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('text', usetex=True)
        
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.imshow(RyR_data2D, interpolation="nearest", origin="lower", \
                   extent=[0,Nx*RyR_size*0.001,0,Ny*RyR_size*0.001], cmap='Greys')
        plt.title(r"$\mathrm{RyRs}$")
        plt.xlabel(r"$\mathrm{\mu{}m}$")
        plt.ylabel(r"$\mathrm{\mu{}m}$")
        plt.subplot(122)
        plt.imshow(jSR_data2D, interpolation="nearest", origin="lower",\
                   extent=[0,Nx*RyR_size*0.001,0,Ny*RyR_size*0.001], cmap='Greys')
        plt.title(r"$\mathrm{jSR\;distribution}$")
        plt.xlabel(r"$\mathrm{\mu{}m}$")
        plt.ylabel(r"$\mathrm{\mu{}m}$")
        plt.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.97, wspace=0.20)
        plt.savefig("{}_ryr_jsr_distr.pdf".format(basename))
        
    t1 = time.time()
    print "Done creating indices: {:.2f}s".format(t1-t0)

    # Double or single precision
    float_type = np.float64 if double else np.float32

    # Save data to file
    double_str = "_double" if double else ""
        
    # Create a geom 3D array first all are cytosol voxels
    geom_vox = np.ones((Nx*subdivide, Ny*subdivide, Nz*subdivide), dtype=np.uint8)
    geom_vox *= inds["cyt"]
    
    # Triple (in each direction) the number of jSR voxels and propagate
    # Find the z index for where the jSR should be inserted
    center_z_ind = Nz/2*subdivide
    for k in range(subdivide):
        geom_vox[:,:,center_z_ind+k] = jSR_data2D*inds["jsr"]

        # Find the tt-indices assume equal thickness...
        tt_offset = 1+cleft_size+k
        geom_vox[:,:,center_z_ind-tt_offset] = jSR_data2D*inds["tt"]

    # Add information about cleft
    geom_vox[:,:,center_z_ind-1] = jSR_data2D*inds["cleft"]

    # Add smoothed top for the jSR
    geom_vox[:,:,center_z_ind+subdivide] = jSR_data2D_top*inds["jsr"]

    # Find RyR faces
    RyR_x_ind, RyR_y_ind = RyR_data.nonzero()
    RyR_x_ind, RyR_y_ind = RyR_x_ind.copy(), RyR_y_ind.copy()

    # Scale and translate indices
    RyR_x_ind*=subdivide
    RyR_y_ind*=subdivide
    RyR_x_ind+=subdivide/2
    RyR_y_ind+=subdivide/2

    # Find the common z index for all RyRs
    RyR_z_ind = np.ones(len(RyR_x_ind))*center_z_ind

    RyRs = np.transpose(np.array((RyR_x_ind, RyR_y_ind, RyR_z_ind, \
                                  RyR_x_ind, RyR_y_ind, RyR_z_ind-1),
                                 dtype=np.uint16))

    # SR indices
    SR_indices = (geom_vox == inds["jsr"]).astype(np.int8)

    # Find SERCA faces in xy plane and in the upper z plane
    # (skipping cleft faces)
    #
    # In 1D we have
    #
    #  SR_indices      == 00111100
    #  SR_indices[:-1] == 0011110
    #  SR_indices[1:]  == 0111100
    #  SR_diff         == 0-100010
    #  x_SERCA will here be:
    #  [2,6]

    # We collect face information by putting two voxels together. The first
    # voxel is the sending one and the next is the recieving one

    # X-direction
    SR_diff_x = SR_indices[:-1,:,:]-SR_indices[1:,:,:]

    # Find indices for x faces that receive and send flux (cyt)
    # FIXME: Switch order so that SERCA go from cyt to SR
    receiving = (SR_diff_x==1).nonzero()
    X_sending_receive_0 = np.transpose(np.array(
        (receiving[0]+1,)+receiving[1:]+receiving))

    sending = (SR_diff_x==-1).nonzero()
    X_sending_receive_1 = np.transpose(np.array(
        (sending+(sending[0]+1,)+sending[1:])))
    
    # Y-direction
    SR_diff_y = SR_indices[:,:-1,:]-SR_indices[:,1:,:]

    # Find indices for y faces that receive and send flux (cyt)
    receiving = (SR_diff_y==1).nonzero()
    Y_sending_receive_0 = np.transpose(np.array(
        (receiving[0],)+(receiving[1]+1,)+(receiving[2],)+receiving))

    sending = (SR_diff_y==-1).nonzero()
    Y_sending_receive_1 = np.transpose(np.array(
        (sending+(sending[0],)+(sending[1]+1,)+(sending[2],))))

    # Z-direction
    SR_diff_z = SR_indices[:,:,:-1]-SR_indices[:,:,1:]

    # Find indices for x faces that receive and send flux (cyt)
    receiving = (SR_diff_z==1).nonzero()
    Z_sending_receive_0 = np.transpose(np.array(
        receiving[:-1]+(receiving[-1]+1,)+receiving))

    sending = (SR_diff_z==-1).nonzero()
    Z_sending_receive_1 = np.transpose(np.array(
        (sending+sending[:-1]+(sending[-1]+1,))))

    SERCA_faces = np.concatenate((X_sending_receive_0, X_sending_receive_1,
                                  Y_sending_receive_0, Y_sending_receive_1,
                                  Z_sending_receive_0))

    # Find SR-cyt face indices in the cleft
    z_cleft = Z_sending_receive_1

    t2 = time.time()
    print "Done generating indices: {:.2f}s".format(t2-t1)
    
    # Save voxel and face data as a h5 file
    with geometry_file("{}{}.h5".format(basename, double_str)) as f:

        f.attrs.create("h", h_geom, dtype=float_type)
        f.attrs.create("global_size", np.array([x_domain_size,
                                                y_domain_size,
                                                z_domain_size],
                                               dtype=float_type))
        g = f.create_group("domains")
        g.attrs.create("num", len(inds), dtype=np.uint8) # cyt, jsr, tt
        for name, ind in inds.items():
            g.attrs.create("name_{}".format(ind), name)
        g.attrs.create("indices", np.array(inds.values(), dtype=np.uint8))
        g.create_dataset("voxels", data=geom_vox, compression="gzip",
                         dtype=np.uint8)

        # Add domain connections
        g.create_dataset("domain_connections", data=connections, dtype=np.uint8)

        faces = f.create_group("boundaries")
        faces.attrs.create("num", 3, dtype=np.uint8) # cyt, jsr, tt
        faces.attrs.create("name_0", "serca")
        faces.attrs.create("name_1", "ryr")
        faces.attrs.create("name_2", "sr_cleft")
        faces.create_dataset("serca", data=SERCA_faces, compression="gzip", \
                             dtype=np.uint16)
        faces.create_dataset("sr_cleft", data=z_cleft, compression="gzip", \
                             dtype=np.uint16)

        faces.create_dataset("ryr", data=RyRs, compression="gzip", \
                             dtype=np.uint16)
        faces.attrs.create("type_0", "density");
        faces.attrs.create("type_1", "discrete");
        faces.attrs.create("type_2", "density");
        
    t3 = time.time()
    print "Done writing to file: {:.2f}s".format(t3-t2)
    
if __name__ == "__main__":

    if 1:
        RyR_data = np.zeros((4,4), dtype=np.bool)
        RyR_data[1,1] = True
        generate_cru_geometry_file(RyR_data, Nz=4, basename="test_small_geometry",
                                   save2d_pdf=True)
        exit()

    RyR_data = np.zeros((14,14), dtype=np.bool)
    RyR_data[2,2] = True
    RyR_data[4,2] = True
    RyR_data[3,4] = True
    RyR_data[2,5] = True
    RyR_data[3,6] = True
    
    RyR_data[7,8]   = True
    RyR_data[8,8]   = True
    RyR_data[10,8]  = True
    RyR_data[9,10]  = True
    RyR_data[11,11] = True

    generate_cru_geometry_file(RyR_data, Nz=12, basename="test_geometry", save2d_pdf=True)
