import argparse
import sys
import matplotlib.pyplot as plt
import time
import h5py
import math
import numpy as np
import glob

directions = dict(x="yz", y="xz", z="xy")
extent_inds = dict(x=[1,2], y=[0,2], z=[0,1])

def args():
    descr = "plot spark propensities from a .h5 file."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("casename", type=str, help="casename for plotting statistics.")
    #parser.add_argument("-ko", "--kd_open", type=float, default=129.92,
    #                    help="Kd for Ca binding for opening a channel.")
    #parser.add_argument("-kox", "--k_max_open", type=float, default=0.7,
    #                    help="Max rate for opening a channel.")
    #parser.add_argument("-kom", "--k_min_open", type=float, default=1.e-4,
    #                    help="Min rate for opening the channel.")
    #parser.add_argument("-no", "--n_open", type=float, default=2.8,
    #                    help="Hill coefficient for opening rate.")
    #parser.add_argument("-kc", "--kd_close", type=float, default=62.5,
    #                    help="Kd for Ca binding for closing a channel.")
    #parser.add_argument("-kcx", "--k_max_close", type=float, default=0.9,
    #                    help="Max rate for closing a channel.")
    #parser.add_argument("-kcm", "--k_min_close", type=float, default=10.,
    #                    help="Min rate for closing a channel.")
    #parser.add_argument("-nc", "--n_close", type=float, default=-0.5,
    #                    help="Hill coefficient for closing rate.")

    return parser

def coupling_fidelity(filenames):

    colors = "brg"

    N_second_ryr_opened = 0
    close_times = []
    min_ca_jsr = []
    max_ca_cleft = []
    
    # Iterate all filenames and find coupling statistics
    for filename in filenames:

        spark_triggered = 0
        with h5py.File(filename) as f:
        
            assert "data_0000000" in f.keys(), "No data in file"
        
            for data_ind, data_name in enumerate(f.keys()):
        
                discrete_ryrs = f[data_name]["discrete_ryr"].value

                # If no ryrs are open we break
                if discrete_ryrs.sum() == 0:
                    if spark_triggered:
                        close_times.append(f[data_name].attrs["time"])
                        min_ca_jsr.append(f[data_name]["jsr"]["Ca"].attrs["min"][0])
                        max_ca_cleft.append(f[data_name]["cleft"]["Ca"].attrs["max"][0])
                    break

                # If more than 1 ryr is open we have a second opening
                if discrete_ryrs.sum() > 1 and not spark_triggered:
                    N_second_ryr_opened += 1
                    spark_triggered = 1

            # If spark did not terminate during simulation
            else:
                close_times.append(f[data_name].attrs["time"])
                min_ca_jsr.append(f[data_name]["jsr"]["Ca"].attrs["min"][0])
                max_ca_cleft.append(f[data_name]["cleft"]["Ca"].attrs["max"][0])

    close_times = np.array(close_times)
    min_ca_jsr = np.array(min_ca_jsr)
    max_ca_cleft = np.array(max_ca_cleft)

    return N_second_ryr_opened*1.0/len(filenames), close_times.mean(), \
           close_times.min(), close_times.max(), min_ca_jsr.mean(), max_ca_cleft.max()
        
if __name__ == "__main__":

    N_RyRs = 57
    center_opening = 1
    filename_glob = "casename-SRxu-h-12-56-28-{}-1-{}-cleft-kdo_105_"\
                    "kdc_62_i_0.5_*-stoch*.h5".format(N_RyRs, center_opening)
    input_files = glob.glob(filename_glob)
    jada = coupling_fidelity(input_files)
    #print jada
    print "N_RyRs {}, Center open: {}, N: {}, Coupling fidelity: {}, Spark"\
          "duration: {:.2f}, [{:.2f}, {:.2f}], min Ca jsr: {:.2f}, "\
          "max Ca cleft: {:.2f}".format(\
        N_RyRs, center_opening, len(input_files), *jada)
    
