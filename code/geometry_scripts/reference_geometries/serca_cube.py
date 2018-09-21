import numpy as np
from collections import OrderedDict
from h5files import GeometryFile

outfile = "test.h5"
double = True

"""Create a 8x8x8 geometry with a 2x2x2 cube of nsr in the center"""

# Voxel labels
inds = OrderedDict()
inds["cyt"] = 0
inds["cleft"] = 1
inds["jsr"] = 2
inds["nsr"] = 3
inds["tt"]  = 4

# Geometry, length unit: nm
nr_of_ryrs = 0
nx = ny = nz = 8
h = 12
domain_size = np.array((nx*h, ny*h, nz*h))
    
domain_connections = [("cyt", "cleft"), ("jsr", "nsr")]
connections = np.zeros((len(inds), len(inds)), dtype=np.uint8)

for conn in domain_connections:   
    assert(len(conn)) == 2, "a connection must be between 2 domains"
    assert conn[0] in inds, "domain %s not recognized" % conn[0]
    assert conn[1] in inds, "domain %s not recognized" % conn[1]
    connections[inds[conn[0]], inds[conn[1]]] = 1
    connections[inds[conn[1]], inds[conn[0]]] = 1

geom_vox = np.empty((nx, ny, nz), dtype=np.uint8)
geom_vox.fill(inds["cyt"])

geom_vox[3:5, 3:5, 3:5] = inds["nsr"]
print (geom_vox == inds["nsr"]).sum()

SERCA_faces = np.transpose((2, 1, 1, 1, 1, 1))[np.newaxis, :]

nsr_indices = (geom_vox == inds["nsr"]).astype(np.int8)

# Find SERCA faces in xy plane and upper z plane
nsr_diff_x = nsr_indices[:-1, :, :] - nsr_indices[1:, :, :]
x, y, z = (nsr_diff_x == 1).nonzero()
x_sending_receive_0_nsr = np.transpose(np.array((x+1, y, z, x, y, z)))
x, y, z = (nsr_diff_x == -1).nonzero()
x_sending_receive_1_nsr = np.transpose(np.array((x, y, z, x+1, y, z)))

nsr_diff_y = nsr_indices[:,:-1,:]-nsr_indices[:,1:,:]
x, y, z = (nsr_diff_y == 1).nonzero()
y_sending_receive_0_nsr = np.transpose(np.array((x, y+1, z, x, y, z)))
x, y, z = (nsr_diff_y == -1).nonzero()
y_sending_receive_1_nsr = np.transpose(np.array((x, y, z, x, y+1, z)))

nsr_diff_z = nsr_indices[:,:,:-1]-nsr_indices[:,:,1:]
x, y, z = (nsr_diff_z == 1).nonzero()
z_sending_receive_0_nsr = np.transpose(np.array((x, y, z+1, x, y, z)))
x, y, z = (nsr_diff_z == -1).nonzero()
z_sending_receive_1_nsr = np.transpose(np.array((x, y, z, x, y, z+1)))

SERCA_faces = np.concatenate((x_sending_receive_0_nsr, x_sending_receive_1_nsr, 
                              y_sending_receive_0_nsr, y_sending_receive_1_nsr, 
                              z_sending_receive_0_nsr, z_sending_receive_1_nsr))

SERCA_faces = np.concatenate((x_sending_receive_0_nsr, x_sending_receive_1_nsr,
                              y_sending_receive_0_nsr, y_sending_receive_1_nsr,
                              z_sending_receive_0_nsr, z_sending_receive_1_nsr))

# Hande double flag
float_type = np.float64 if double else np.float32

# Save voxel and face data to a h5 file
with GeometryFile(outfile) as f:
    f.attrs.create("h", h, dtype=float_type)
    f.attrs.create("global_size", domain_size.astype(float_type))

    g = f.create_group("domains")
    g.attrs.create("num", len(inds), dtype=np.uint8)
        
    for name, ind in inds.items():
        g.attrs.create("name_{}".format(ind), name)
            
    g.attrs.create("indices", np.array(inds.values(), dtype=np.uint8))
    g.create_dataset("voxels", data=geom_vox, compression="gzip",
                     dtype=np.uint8)

    # Add domain connections
    g.create_dataset("domain_connections", data=connections, dtype=np.uint8)
    
    faces = f.create_group("boundaries")
    faces.attrs.create("num", 3, dtype=np.uint8)
    faces.attrs.create("name_0", "serca")
    faces.attrs.create("name_1", "ryr")
    faces.attrs.create("name_2", "sr_cleft")

    faces.create_dataset("serca", data=SERCA_faces, compression="gzip",
                         dtype=np.uint16)
    faces.create_dataset("sr_cleft", data=np.array(()), compression="gzip",
                            dtype=np.uint16)
    faces.create_dataset("ryr", data=np.array(()), compression="gzip",
                            dtype=np.uint16)
                             
    faces.attrs.create("type_0", "density")
    faces.attrs.create("type_1", "discrete")
    faces.attrs.create("type_2", "density")