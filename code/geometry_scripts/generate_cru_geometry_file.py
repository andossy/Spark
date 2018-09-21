import time
import logging
import numpy as np 
import matplotlib.pyplot as plt
from collections import OrderedDict
import os, sys

import geometry_util as gu
from h5files import GeometryFile

# Voxel labels
INDS = OrderedDict()
INDS["cyt"] = 0
INDS["cleft"] = 1
INDS["jsr"] = 2
INDS["nsr"] = 3
INDS["tt"]  = 4

# CreateCONNECTIONS
domain_connections = [("cyt", "cleft"), ("jsr", "nsr")]
CONNECTIONS = np.zeros((len(INDS), len(INDS)), dtype=np.uint8)
for conn in domain_connections:
    assert(len(conn)) == 2, "a connection must be between 2 domains"
    assert conn[0] in INDS, "domain %s not recognized" % conn[0]
    assert conn[1] in INDS, "domain %s not recognized" % conn[1]
    CONNECTIONS[INDS[conn[0]], INDS[conn[1]]] = 1
    CONNECTIONS[INDS[conn[1]], INDS[conn[0]]] = 1



def ppm_to_arrays(infile, width, jsr_layers, subdivide=3, nsr_width=1, contigious_jsr=False):
    """Read a ppm and create boolean arrays for ryr, jsr and nsr.

    A .ppm is read, assuming a resolution of 36 nm. From these 
    RyR locations, the jSR is created by selecting the area surrounding RyR.
    The RyR and jSR arrays are then refined to a finer grid. Finally the nSR
    column locations are selected by surrounding the refined RyR locations

    :param str infile: name of .ppm file (with or without extension)
    :param int width: number of voxels in each direction [36 nm]
    :param int subdivide: number of subpixels to split each pixel into
    :param int jsr_layers: nr of layers of jSR around RyR
    :param nsr_width: size of nSR columns around RyR
    """
    # Read RyR data and center it to correct size
    indata = gu.read_ppm(infile)
    bbox = gu.trim_to_bbox(indata)
    ryr = gu.pad_array(bbox, (width, width))

    # Create jSR
    jsr = gu.create_jsr(ryr, jsr_layers)

    # Refine RyR and jSR
    ryr = gu.refine_ryr(ryr, subdivide)
    jsr = gu.refine_jsr(jsr, subdivide, remove_edges=True)
    nsr = gu.create_nsr_columns(ryr)

    if contigious_jsr is True:
        jsr = gu.contigious_jsr(jsr, ryr)
    # Find nSR column locations
    return ryr, jsr, nsr

def generate_cru_geometry_file(ryr, jsr, nsr, ttub=None, ttub_mode='convexhull', ttubpad=14, 
                               nz=168, 
                               nsr_height=None, sr_volume=None,
                               cleft_size=1, jsr_thickness=3,
                               double=True, basename="sarcomere_geometry"):
    """Generate a 3D voxel geometry from 2D arrays.

    Take RyR, jSR and nSR positions as 2D boolean arrays and produce
    a 3D voxel geometry that is saved to a h5 file. Each voxel is
    12x12x12 microns, and the 2D arrays are expected to have the same 
    resolution. 

    The height of the nSR columns can be given either as the height, in number 
    of voxels, or as the volume of the nSR, either in absolute value, or as
    a ratio of total volume.

    :param ryr: (m, n) boolean array
    :type ryr: 
    :param jsr: (m, n) boolean array 
    :param nsr: (m, n) boolean array 
    :param nz: int, number of voxels in z-direction [12 nm]
    :param nsr_height: height of nsr in voxels, assumes height is max if None
    :param nsr_volume: volume of nsr in voxels
    :param nsr_ratio: ratio of nsr to total volume
    :param cleft_size: int, width of cleft in voxels
    :param jsr_thickness: int, jSR thickness in z-iredction in voxels
    :param double: boolean, save to h5 with double float precision
    :param basename: name of h5 file to save to 
    """
    assert all((len(data.shape) == 2) for data in (ryr, jsr, nsr)), "expected "\
                                                                    "2D arrays"
    assert ryr.shape == jsr.shape == nsr.shape, "2D arrays must be of same size"

    single_str = "_single" if not double else ""
    basename = basename.replace('.h5', '')
    outfile = "{}{}.h5".format(basename, single_str)
    logging.info("Creating CRU geometry file {}".format(outfile))

    # Geometry, length unit: nm
    nr_of_ryrs = ryr.sum()
    h = 12
    nx, ny = ryr.shape
    domain_size = np.array((nx*h, ny*h, nz*h))
    
   
    # Create a geometric 3D array, start all voxels as cytosol
    geom_vox = np.empty((nx, ny, nz), 
                         dtype=np.uint8)
    geom_vox.fill(INDS["cyt"])

    # Find z indices
    center_z = nz/2
    last_z_ind = nz
    
    # Fill in jSR and t-tubule voxels (assuming equal thickness of jSR and tt)
    # Assume t-tubules is opposite jSR if no information is given
    if ttub is None:
        if ttub_mode == 'copy':
            ttub = jsr.copy()
        elif ttub_mode in ['convex', 'convexhull', 'convex_hull']:
            ttub = gu.create_ttub_from_ryr(ryr, mode=ttub_mode, pad=ttubpad)
        else:
            raise ValueError("Ttub mode not understood, valid are copy/convexhull.")

    for k in range(jsr_thickness):
        geom_vox[:, :, center_z + k] = jsr*INDS["jsr"]
        geom_vox[:, :, center_z - k - cleft_size - 1] = ttub*INDS['tt']

    # Smooth the top layer of the jSR
    geom_vox[:,:,center_z+jsr_thickness-1] = gu.shrink_sr(jsr, 1)*INDS["jsr"] 
    jsr_vol = (geom_vox == INDS["jsr"]).sum()

    # Find the nSR roof
    roof_height = 2
    x, y = np.nonzero(nsr)
    x0, y0, x1, y1 = (min(x), min(y), max(x), max(y))
    roof_volume = (x1 - x0)*(y1 - y0)*roof_height
    
    # Calculate volume of nSR columns
    total_vol = np.prod(geom_vox.shape)
    if all(arg is None for arg in (nsr_height, sr_volume)):
        sr_volume = 28935

    if nsr_height is None:
        col_volume = sr_volume - jsr_vol - roof_volume
        col_area = float(np.sum(nsr))
        nsr_height = int(np.ceil(col_volume/col_area))
        
        if nsr_height > nz/2 - jsr_thickness - roof_height:
            nsr_height = nz/2 - jsr_thickness - roof_height

        roof_width_increase = 0
        roof_height_increase = 0
        while nsr_height*col_area + roof_volume + jsr_vol < sr_volume:
            if roof_width_increase < 15:            
               x0 -= 1; y0 -= 1; x1 += 1; y1 += 1
               roof_width_increase += 1
            elif roof_height_increase < 10:
                roof_height += 1
                roof_height_increase += 1
            else: 
                break

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(x1, geom_vox.shape[0])
            y1 = min(y1, geom_vox.shape[1])
            roof_volume = (x1 - x0)*(y1 - y0)*roof_height

    if nsr_height < 2:
        logging.warning("Specified nSR volume too small")
        nsr_height = 2

    nsr_stop_ind = center_z + jsr_thickness + nsr_height

    if nsr_stop_ind > nz - roof_height:
        logging.warning("Specified nSR volume does not fit inside geometry."
                        "({}>{})".format(nsr_stop_ind, nz - roof_height))

        nsr_stop_ind = nz - roof_height

    # Add columns
    for m in range(center_z + jsr_thickness, nsr_stop_ind):
        geom_vox[:,:,m] = nsr*INDS["nsr"]
        
    # Add roof
    geom_vox[x0:(x1+1), y0:(y1+1), nz-roof_height::] = INDS["nsr"]
    # Print volume information
    jsr_vol = (geom_vox == INDS["jsr"]).sum()
    nsr_vol = (geom_vox == INDS["nsr"]).sum()
    logging.info("{:s}".format("Volume measurements (number of voxels)"))
    logging.info("{:>15s}: {:<10.0f}".format("Total volume", total_vol))
    logging.info("{:>15s}: {:<10.0f} ({:.2%})".format("jsr volume", jsr_vol, 
                                              1.0*jsr_vol/total_vol))
    logging.info("{:>15s}: {:<10.0f} ({:.2%})".format("nsr volume", nsr_vol, 
                                              1.0*nsr_vol/total_vol))
    logging.info("Total sr volume of 1 micron cube: {:.2%}".format(1.0*(jsr_vol+nsr_vol)/578703.7))

    # Fill in cleft voxels
    for k in range(cleft_size):
        geom_vox[:, :, center_z-k-1] = jsr*INDS["cleft"]

    # Construct 3D RyR indices
    ryr_x_ind, ryr_y_ind = ryr.nonzero()
    ryr_z_ind = np.empty(nr_of_ryrs, dtype=np.uint16)
    ryr_z_ind.fill(center_z)

    ryrs = np.transpose(np.array((ryr_x_ind, ryr_y_ind, ryr_z_ind, \
                                  ryr_x_ind, ryr_y_ind, ryr_z_ind-1),
                                 dtype=np.uint16))

    # Create LCCs (At the moment we only model sparks, so no LCCs)
    # create_LCCs()

    # SR indices
    jsr_indices = (geom_vox == INDS["jsr"]).astype(np.int8)
    nsr_indices = (geom_vox == INDS["nsr"]).astype(np.int8)

    # Find SERCA faces in xy plane and upper z plane (skipping cleft faces)
    # Find indices for x faces that recieve and send flux (cyt)
    # FIXME: Switch order so that SERCA goes from cyt to SR
    jsr_diff_x = jsr_indices[:-1, :, :] - jsr_indices[1:, :, :]
    nsr_diff_x = nsr_indices[:-1, :, :] - nsr_indices[1:, :, :]
    # Recieving in x
    x, y, z = (jsr_diff_x == 1).nonzero()
    x_sending_receive_0_jsr = np.transpose(np.array((x+1, y, z, x, y, z)))
    x, y, z = (nsr_diff_x == 1).nonzero()
    x_sending_receive_0_nsr = np.transpose(np.array((x+1, y, z, x, y, z)))
    # Sending in x
    x, y, z = (jsr_diff_x == -1).nonzero()
    x_sending_receive_1_jsr = np.transpose(np.array((x, y, z, x+1, y, z)))
    x, y, z = (nsr_diff_x == -1).nonzero()
    x_sending_receive_1_nsr = np.transpose(np.array((x, y, z, x+1, y, z)))

    # Find indices for y-faces
    jsr_diff_y = jsr_indices[:,:-1,:]-jsr_indices[:,1:,:]
    nsr_diff_y = nsr_indices[:,:-1,:]-nsr_indices[:,1:,:]
    # Recieving in y
    x, y, z = (jsr_diff_y == 1).nonzero()
    y_sending_receive_0_jsr = np.transpose(np.array((x, y+1, z, x, y, z)))
    x, y, z = (nsr_diff_y == 1).nonzero()
    y_sending_receive_0_nsr = np.transpose(np.array((x, y+1, z, x, y, z)))
    # Sending in y
    x, y, z = (jsr_diff_y == -1).nonzero()
    y_sending_receive_1_jsr = np.transpose(np.array((x, y, z, x, y+1, z)))
    x, y, z = (nsr_diff_y == -1).nonzero()
    y_sending_receive_1_nsr = np.transpose(np.array((x, y, z, x, y+1, z)))

    # Find indices for z-faces
    jsr_diff_z = jsr_indices[:,:,:-1] - jsr_indices[:,:,1:]
    nsr_diff_z = nsr_indices[:,:,:-1] - nsr_indices[:,:,1:]
    
    # Recieving in Z
    x, y, z = (jsr_diff_z == 1).nonzero()
    # Create xy tuples
    xy_serca_INDS = zip(*(x, y))
    include_INDS = range(len(xy_serca_INDS))
    # Remove indices which coincides with the nsr xy indices
    for xy_nsr_loc in zip(*nsr.nonzero()):
        include_INDS.remove(xy_serca_INDS.index(xy_nsr_loc))
    z_sending_receive_0_jsr = np.transpose(np.array((x, y, z + 1, x, y, z)))
    z_sending_receive_0_jsr = z_sending_receive_0_jsr[include_INDS]

    # Sending in Z
    x, y, z = (nsr_diff_z == -1).nonzero()
    z_sending_receive_1_nsr = np.transpose(np.array((x, y, z, x, y, z + 1)))
    # Split all xy indices into a list of xy tuples
    xy_serca_INDS = zip(*(x, y))
    include_INDS = range(len(xy_serca_INDS))
    # Remove indices which coincides with the nsr xy indices
    for xy_nsr_loc in zip(*nsr.nonzero()):
        include_INDS.remove(xy_serca_INDS.index(xy_nsr_loc))
    z_sending_receive_1_nsr = z_sending_receive_1_nsr[include_INDS]

    SERCA_faces = np.concatenate((x_sending_receive_0_jsr, x_sending_receive_1_jsr, 
                                  x_sending_receive_0_nsr, x_sending_receive_1_nsr, 
                                  y_sending_receive_0_jsr, y_sending_receive_1_jsr, 
                                  y_sending_receive_0_nsr, y_sending_receive_1_nsr, 
                                  z_sending_receive_0_jsr, z_sending_receive_1_nsr))

    # Find SR-cyt face indices in the cleft
    x, y, z = (jsr_diff_z == -1).nonzero()
    z_cleft = np.transpose(np.array((x, y, z, x, y, z + 1)))

    # Hande double flag
    float_type = np.float64 if double else np.float32
    write_geomfile(outfile, geom_vox, SERCA_faces, z_cleft, ryrs, double=double)
    

def write_geomfile(outfile, geom_vox, serca, cleft, ryr, h=12, double=True):
    float_type = np.float64 if double is True else np.float32

    nx, ny, nz = geom_vox.shape
    domain_size = np.array((nx*h, ny*h, nz*h), dtype=float_type)

    with GeometryFile(outfile) as f:
        f.attrs.create("h", h)
        f.attrs.create("global_size", domain_size)

        # Set up domain indicators
        g = f.create_group("domains")
        g.attrs.create("num", len(INDS), dtype=np.uint8)
        for name, ind in INDS.items():
            g.attrs.create("name_{}".format(ind), name)
        g.attrs.create("indices", np.array(INDS.values(), dtype=np.uint8))

        # Fill in voxel data
        g.create_dataset("voxels", data=geom_vox, compression="gzip",
                         dtype=np.uint8)

        # Set up domain connections and boundaries
        g.create_dataset("domain_connections", data=CONNECTIONS, dtype=np.uint8)
        faces = f.create_group("boundaries")
        faces.attrs.create("num", 3, dtype=np.uint8)
        faces.attrs.create("name_0", "serca")
        faces.attrs.create("name_1", "ryr")
        faces.attrs.create("name_2", "sr_cleft")

        faces.create_dataset("serca", data=serca, compression="gzip",
                             dtype=np.uint16)
        faces.create_dataset("sr_cleft", data=cleft, compression="gzip",
                             dtype=np.uint16)
        faces.create_dataset("ryr", data=ryr, compression="gzip",
                             dtype=np.uint16)

        faces.attrs.create("type_0", "density")
        faces.attrs.create("type_1", "discrete")
        faces.attrs.create("type_2", "density")
        

def ppm_to_geomfile(infile, outfile, pad, N=56, ttub_mode='convexhull', ttubpad=14,
                    save2d_pdf=True, contigious_jsr=False, **kwargs):
    """Read a ppm file and generate a 2x2x2 um geometry h5 file."""
    ryr, jsr, nsr = ppm_to_arrays(infile, N, pad, contigious_jsr=contigious_jsr)

    ttub = gu.create_ttub_from_ryr(ryr, mode=ttub_mode, pad=ttubpad)

    if save2d_pdf:
        import os
        gu.save_combined_2D_pdf(os.path.split(outfile)[1], ryr, jsr, nsr)

    generate_cru_geometry_file(ryr, jsr, nsr, tt=ttub, nz=3*N, cleft_size=1, 
                               jsr_thickness=3, double=True, basename=outfile,
                               **kwargs)

        

if __name__ == '__main__':
    import argparse
   
    logging.getLogger().setLevel(0)
    parser = argparse.ArgumentParser(description='Create a .h5 geometry from '
                                                 'RyR locations.',
                                     usage='%(prog)s ppm_file padding [options]'
                                           '\nUse --help for more info.' )

    parser.add_argument('ppm_file', type=str, 
                        help='path to .ppm-file of RyR locations')

    parser.add_argument('padding', type=int,
                        help='Layers of jSR to pad around each RyR')

    parser.add_argument('--width', type=int, metavar='N', default=56,
                        help='Width of geometry in 36 micron voxels')

    parser.add_argument('--height', type=int, metavar='nz', default=0,
                        help='Height of geometry in 36 micron voxels')

    parser.add_argument('--save2d_pdf', dest='pdf', action='store_true',
                         help='Save a pdf of 2D RyR, jSR and nSR layers')

    parser.add_argument('--single', dest='double', action='store_false',
                        help='Create a .h5 file with single float precision')

    parser.add_argument('--double', dest='double', action='store_true',
                        help='Create a .h5 file with double float precision')

    parser.add_argument('--outfile', type=str, default=None,
                         help='Path to store .h5 file')

    parser.add_argument('--contigious_jsr', dest='contigious_jsr', action='store_true',
                        help='Makes jSR contigious by closing of RyR.')

    args = parser.parse_args()

    outfile = args.outfile
    if outfile is None:
        outfile = "geometries/P%d_" % args.padding+os.path.split(args.ppm_file)[-1]
        outfile = outfile.replace('.ppm', '')
    else:
        outfile = outfile.replace('.h5' , '')
    
    pdf = False if args.pdf is None else True
    double = False if args.double is False else True

    ryr, jsr, nsr = ppm_to_arrays(args.ppm_file, args.width, args.padding, contigious_jsr=args.contigious_jsr)

    if pdf is not None and pdf:
        import os
        gu.save_combined_2D_pdf(os.path.split(outfile)[1], ryr, jsr, nsr)

    height = 3*args.width if args.height is 0 else 3*args.height

    generate_cru_geometry_file(ryr, jsr, nsr, nz=height, cleft_size=1, 
                               jsr_thickness=3, double=double, basename=outfile)

    # generate_cru_geometry_file(ryr, jsr, nsr, nz=3*args.height, cleft_size=1, 
    #                            jsr_thickness=3, double=double, basename=outfile,
    #                            nsr_ratio=0.005)

