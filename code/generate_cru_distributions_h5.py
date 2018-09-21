#!/bin/python
# -*- coding: utf-8 -*-
import h5py
import os
import shutil
import time
from CRU_generation import growth_model, pylab, output
import numpy as np
np.random.seed(2)

float_type = np.float32
float_type = np.float64

class geometry_file:
    def __init__(self, filename):
        self.old_file = None
        self.filename = filename
    
    def __enter__(self):

        # If filename already exist
        if os.path.isfile(self.filename):
            self.old_file = "_old_"+self.filename
            shutil.move(self.filename, self.old_file)
        
        # Open a h5 file
        self.f = h5py.File(self.filename)
        return self.f 

    def __exit__(self, type, value, traceback):
        self.f.close()
        if type is None:
            if self.old_file:
                os.unlink(self.old_file)
        else:
            shutil.move(self.old_file, self.filename)
        
# Geometry related parameters
xy_domain_size = 5 # um
z_domain_size = 1  # um

h_nm = 36
h_um = h_nm * 0.001
Nxy = xy_domain_size*1000/h_nm
Nz = z_domain_size*1000/h_nm

h_geom = 12
cleft_hight = 12
cleft_size = cleft_hight/h_geom

assert h_nm % h_geom == 0

# Voxel indices
Cyt_ind = 0
jSR_ind = 1
TT_ind = 2

# Face indices
SERCA_ind = 10
RyR_ind = 20

subdivide = h_nm / h_geom

# RyR generation parameters
iterations=150
RyR_cutoff = 3
P_growth=1.4e-2      # Healthy 1.40e-2
num_sarcomeres = 1
# Healthy stats:
# Mean size:  13.9 ± 14.0 [3.0, 85.0]
# Mean closest distance: 222.7 ± 121.6 [90.0, 721.2]
# Mean cluster radius: 242.5 ± 288.3 [24.5, 2174.0]

# Set the relative coordinates of who are neighbors
diagonals = [[1,1], [1,-1], [-1,1], [-1,-1]]
neighbors = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]
neighbors_2nd = [[2,2], [2,1], [2,0], [2,-1], [2,-2], \
                 [-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2],\
                 [0,2], [0,-2], [1,2], [1,-2], [-1,2], [-1,-2]]

# Add all diagonals to neigbors
neighbors += diagonals

# Also add all 2nd neighbors
neighbors += neighbors_2nd

#P_vec = [[0.125e-4, 0.96], # Healthy [0.5e-4, 0.937]
#         [10.e-4, 0.892]] # Failing [2.5e-4, 0.915]
P_vec = [[0.5e-4, 0.937],
         [2.5e-4, 0.915]]

base_size = 6
fig=pylab.figure(figsize=(base_size*2, base_size*len(P_vec)))
pylab.interactive(False)

do_plot = False
cmap = pylab.get_cmap("binary")
for ind_sarcomere in range(num_sarcomeres):
    pylab.clf()
    for ind_failure, (P_nucleation, P_retention) in enumerate(P_vec):
        
        # Compute the CRU clusters
        t0 = time.time()
        RyR_data, jSR_data, neighbor_indices, clusters, centers, sizes, cluster_distances = \
              growth_model(iterations, Nxy, P_nucleation, P_growth, P_retention, RyR_cutoff,
                           neighbors)
        if ind_failure==1:
            output(clusters, cluster_distances, sizes, "failing")
        else:
            output(clusters, cluster_distances, sizes)
        t1 = time.time()
        print "Done generating RyR CRUs: {:.2f}s".format(t1-t0)

        if do_plot:
            pylab.figure(fig.number)
            pylab.subplot(len(P_vec), 2 ,1+ind_failure*len(P_vec))
            pylab.imshow(RyR_data, interpolation="nearest", origin="lower",\
                         extent=[0,Nxy*h_um,0,Nxy*h_um], cmap=cmap)
            title = "RyR CRUs sarc: {}".format(ind_sarcomere)
            if ind_failure:
                title += " (failing)"
            pylab.title(title)
            pylab.xlabel("$\mu$m")
            pylab.ylabel("$\mu$m")
            
            pylab.figure(fig.number)
            pylab.subplot(len(P_vec), 2, 2+ind_failure*(len(P_vec)))
            pylab.imshow(jSR_data, interpolation="nearest", origin="lower",\
                         extent=[0,Nxy*h_um,0,Nxy*h_um], cmap=cmap)
            title = "JSR distribution sarc: {}".format(ind_sarcomere)
            if ind_failure:
                title += " (failing)"
            pylab.title(title)
            pylab.xlabel("$\mu$m")
            pylab.ylabel("$\mu$m")
    
        # Save indices to file
        failing = "" if ind_failure == 0 else "_failing"
        double = "" if float_type == np.float32 else "_double"
        
        # Tripple the number of jSR voxels and propagate
        # Create a geom 3D array 
        geom_vox = np.ones((Nxy*subdivide, Nxy*subdivide, Nz*subdivide), dtype=np.uint8)
        geom_vox *= Cyt_ind
        
        # Find the z index for where the jSR should be inserted
        center_z_ind = Nz/2*subdivide
        for i in range(subdivide):
            for j in range(subdivide):
                for k in range(subdivide):
                    geom_vox[i::subdivide,j::subdivide,center_z_ind+k] = jSR_data*jSR_ind

                    # Find the tt-indices assume equal thickness...
                    tt_offset = 1+cleft_size+k
                    geom_vox[i::subdivide,j::subdivide,center_z_ind-tt_offset] = \
                                                        jSR_data*TT_ind
        
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
        SR_indices = (geom_vox == jSR_ind).astype(np.int8)

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
        receiving = (SR_diff_x==-1).nonzero()
        X_sending_receive_0 = np.transpose(np.array(
            (receiving[0]+1,)+receiving[1:]+receiving))

        sending = (SR_diff_x==1).nonzero()
        X_sending_receive_1 = np.transpose(np.array(
            (sending+(sending[0]+1,)+sending[1:])))
        
        # Y-direction
        SR_diff_y = SR_indices[:,:-1,:]-SR_indices[:,1:,:]

        # Find indices for y faces that receive and send flux (cyt)
        receiving = (SR_diff_y==-1).nonzero()
        Y_sending_receive_0 = np.transpose(np.array(
            (receiving[0],)+(receiving[1]+1,)+(receiving[2],)+receiving))

        sending = (SR_diff_y==1).nonzero()
        Y_sending_receive_1 = np.transpose(np.array(
            (sending+(sending[0],)+(sending[1]+1,)+(sending[2],))))

        # Z-direction
        SR_diff_z = SR_indices[:,:,:-1]-SR_indices[:,:,1:]

        # Find indices for x faces that receive and send flux (cyt)
        receiving = (SR_diff_z==-1).nonzero()
        Z_sending_receive_0 = np.transpose(np.array(
            receiving[:-1]+(receiving[-1]+1,)+receiving))

        sending = (SR_diff_z==1).nonzero()
        Z_sending_receive_1 = np.transpose(np.array(
            (sending+sending[:-1]+(sending[-1]+1,))))

        SERCA_faces = np.concatenate((X_sending_receive_0, X_sending_receive_1,
                                      Y_sending_receive_0, Y_sending_receive_1,
                                      Z_sending_receive_1))

        # Find SR-cyt face indices in the cleft
        z_cleft = Z_sending_receive_0

        t2 = time.time()
        print "Done generating indices: {:.2f}s".format(t2-t1)
        
        # Save voxel and face data as a h5 file
        with geometry_file("sarcomere_geometry{}{}.h5".format(failing, double)) as f:

            f.attrs.create("h", h_geom, dtype=float_type)
            f.attrs.create("global_size", np.array([xy_domain_size*1000,
                                                    xy_domain_size*1000,
                                                    z_domain_size*1000],
                                                   dtype=float_type))
            g = f.create_group("domains")
            g.attrs.create("num", 3, dtype=np.uint8) # cyt, jsr, tt
            g.attrs.create("name_0", "cyt")
            g.attrs.create("name_1", "jsr")
            g.attrs.create("name_2", "tt")
            g.attrs.create("indices", np.array([Cyt_ind, jSR_ind, TT_ind], dtype=np.uint8))
            #g["names"] = np.array(["cyt", "jsr", "tt"])
            #g["indices"] = np.array([Cyt_ind, jSR_ind, TT_ind], dtype=np.uint8)
            g.create_dataset("voxels", data=geom_vox, compression="gzip",
                             dtype=np.uint8)
            
            faces = f.create_group("boundaries")
            faces.attrs.create("num", 3, dtype=np.uint8) # cyt, jsr, tt
            faces.attrs.create("name_0", "serca")
            faces.attrs.create("name_1", "ryr")
            faces.attrs.create("name_2", "sr_cleft")
            #faces["names"] = np.array(["serca", "ryr", "sr_cleft"])
            faces.create_dataset("serca", data=SERCA_faces, compression="gzip", \
                                 dtype=np.uint16)
            faces.create_dataset("sr_cleft", data=z_cleft, compression="gzip", \
                                 dtype=np.uint16)

            faces.create_dataset("ryr", data=RyRs, compression="gzip", \
                                 dtype=np.uint16)
            
        t3 = time.time()
        print "Done writing to file: {:.2f}s".format(t3-t2)
        #print z_SERCA
        #print z_RyR
        #print z_cleft
        #
        #print "num RyR in cleft:", sum(ryr_ind in z_cleft for ryr_ind in z_RyR)
        #print "num jSR_orig", jSR_data.sum()
        #print "num jSR", (geom_vox==jSR_ind).sum()
        #print "num TT", (geom_vox==TT_ind).sum()
        #print "num SERCA faces", len(x_SERCA), len(y_SERCA), len(z_SERCA)
        
        #data.nonzero()[0].tofile(open("i_RyR_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
        #data.nonzero()[1].tofile(open("j_RyR_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
        #csqn_data.nonzero()[0].tofile(open("i_csqn_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
        #csqn_data.nonzero()[1].tofile(open("j_csqn_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
    
    # Output statistics
    #output(clusters, cluster_distances)
    #pylab.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.95, wspace=0.04, hspace=0.13)
    #pylab.draw()
    #f.savefig("cluster_distribution_sarc_{}.pdf".format(ind_sarcomere))

pylab.show()
    
    
