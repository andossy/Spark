import h5py
import os
import sys
import subprocess
from generate_test_geometries import geom_sizes, N_RyRs

def run_calcium_runs(geom_filename, species_filename, np=0, tstop=50, tsave=1.0, h=6,
                     output_sheets=None, deterministic=True, open_ryrs=None,
                     z_coords=None, casename=None):

    open_ryrs = open_ryrs or []
    output_sheets = output_sheets or []
    z_coords = z_coords or []

    args = ["mpirun", "-np {}".format(np), "./calcium_sparks"] \
           if np>0 else ["./calcium_sparks"]
    args.append("-g {}".format(geom_filename))
    args.append("-m {}".format(species_filename))
    args.append("-t {}".format(tstop))
    args.append("-h {}".format(h))
    args.append("-v")
    args.append("-d {}".format(tsave))
    if z_coords:
        args.append("-z {}".format(" ".join(str(z_coord) for z_coord in z_coords)))
    if output_sheets:
        args.append("-s {}".format(" ".join(output_sheets)))
    if open_ryrs:
        args.append("-R {}".format(" ".join(str(o) for o in open_ryrs)))
    if deterministic:
        args.append("-C {}".format(tstop))
    if casename is not None:
        args.append("-c {}".format(casename))
        
    print args
    return subprocess.Popen(" ".join(args), shell=True)

if __name__ == "__main__":

    species_filename = "parameters_double.h5"
    tstop = 50.
    tsave = 1.
    h = 4.
    output_sheets = ["Ca"]
    for (xy, z), N_RyR in zip(geom_sizes, N_RyRs):
        geom_filename = "test_geometry_xy_{}_z_{}_N_RyR_{}_double.h5".format(xy, z, N_RyR)
        casename = "test_xy_{}_z_{}_N_RyR_{}".format(xy, z, N_RyR)
        f = h5py.File(geom_filename)
        z_coord = f.attrs["h"]*f["boundaries"]["ryr"][-1][-1]
        
        run_calcium_runs(geom_filename, species_filename, np=4, tstop=50, tsave=1.0, h=h,
                         output_sheets=["Ca"], z_coords=[z_coord],
                         deterministic=True, open_ryrs=range(N_RyR),
                         casename=casename).wait()
        
