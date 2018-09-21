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
    parser.add_argument("ifile", type=str, help="HDF5 file .h5 filename.")
    parser.add_argument("-ko", "--kd_open", type=float, default=129.92,
                        help="Kd for Ca binding for opening a channel.")
    parser.add_argument("-kox", "--k_max_open", type=float, default=0.7,
                        help="Max rate for opening a channel.")
    parser.add_argument("-kom", "--k_min_open", type=float, default=1.e-4,
                        help="Min rate for opening the channel.")
    parser.add_argument("-no", "--n_open", type=float, default=2.8,
                        help="Hill coefficient for opening rate.")
    parser.add_argument("-kc", "--kd_close", type=float, default=62.5,
                        help="Kd for Ca binding for closing a channel.")
    parser.add_argument("-kcx", "--k_max_close", type=float, default=0.9,
                        help="Max rate for closing a channel.")
    parser.add_argument("-kcm", "--k_min_close", type=float, default=10.,
                        help="Min rate for closing a channel.")
    parser.add_argument("-nc", "--n_close", type=float, default=-0.5,
                        help="Hill coefficient for closing rate.")

    return parser

def plot_rates(args, center=1, first=0, kd_open_range=[80, 130, 3], dt=0.1):

    kd_open_range = np.linspace(*kd_open_range)
    def close_rate(Ca):
        return min(max((Ca/args.kd_close)**args.n_close, args.k_min_close), args.k_max_close)
    
    def open_rate(Ca):
        return min(max((Ca/args.kd_open)**args.n_open, args.k_min_open), args.k_max_open)
    
    assert "-1-0-" in args.ifile, "expected perifery RyR opening"
    if center:
        args.ifile = args.ifile.replace("-1-0-", "-1-1-")

    print args.ifile
    colors = "brg"

    with h5py.File(args.ifile) as f:
        h = int(f.attrs["h"])
        N = f.attrs["N"]
        geometry_file = f.attrs["geometry_file"]

        # Open geometry file
        with h5py.File(geometry_file) as fg:

            h_geom = int(fg.attrs["h"])
            subdivisions = h/h_geom
            offset = subdivisions/2
            
            # Get ryrs
            ryrs_inds = []
            z_ind = -1

            for ryr_inds in fg["boundaries"]["ryr"]:
                if z_ind == -1:
                    z_ind = ryr_inds[-1]*subdivisions+offset
                else:
                    assert z_ind == ryr_inds[-1]*subdivisions+offset, \
                           "RyRs not in the same z layer"

                # Pick first and second ind meaning we only consider
                # xy ryrs
                ryrs_inds.append((ryr_inds[0]*subdivisions+offset, \
                                  ryr_inds[1]*subdivisions+offset))

            N_ryrs = fg["boundaries"]["ryr"].shape[0]
            
        assert "data_0000000" in f.keys(), "No data in file"
        sheet_names = [data_key for data_key in f["data_0000000"].keys()
                       if "sheet" in data_key]
        assert sheet_names, "No sheets stored in file."

        #print sheet_names
        species_names = sorted(set(name.split("_sheet_")[0] for name in sheet_names))
        
        # Get sheet direction
        sheet_planes = [species[-1] for species in species_names]
        species_names = [name[:-2] for name in species_names]
        assert "Ca" in species_names, "expected Ca to be in data"
        
        indices = sorted(set(int(name.split("_sheet_")[1]) for name in sheet_names))
        #print indices, z_ind, sheet_planes
        assert z_ind in indices, "expected z_ind of RyRs to be in the registered z sheet"
        assert sheet_planes[indices.index(z_ind)] == "z", \
               "expected the sheet plane to be in the z direction"

        # Only pick ryr z ind
        indices = [z_ind]
        
        #base_scale = 6
        #figsize = [base_scale*1.5, base_scale]
        #plt.rcParams.update({'figure.figsize': figsize})
        #fig = plt.figure()
        #plt.clf()
        time_data = []
        ryr_ca = [[] for _ in ryrs_inds]
        #ryr_k_open = []
        #ryr_k_close = []
        Ca_open = []
        P_spark = [[] for _ in kd_open_range]
        k_spark = [[] for _ in kd_open_range]
        T_spark = [[] for _ in kd_open_range]
        p_close_int = [0. for _ in kd_open_range]
        p_open_int = [0. for _ in kd_open_range]
        P_no_spark = [[] for _ in kd_open_range]
        P_cum_spark = [[] for _ in kd_open_range]
        p_something_happen = [[0. for _ in range(N_ryrs)] for __ in kd_open_range]
        center_str = "; center" if center else ""
        ls = "-" if center else "--"
        all_data = f.keys()
        for data_ind in range(len(all_data)-1):
            #ryr_k_open.append(0.)
            #ryr_k_close.append(0.)
            data_name_0 = all_data[data_ind]
            data_name_1 = all_data[data_ind+1]
            t_0 = f[data_name_0].attrs["time"][-1]
            t_1 = f[data_name_1].attrs["time"][-1]
            t = (t_0+t_1)/2
            discrete_ryrs = f[data_name_0]["discrete_ryr"]
            time_data.append(t)
            #print "time:", t
            data_0 = f[data_name_0]["Ca_z_sheet_{}".format(z_ind)].value
            data_1 = f[data_name_1]["Ca_z_sheet_{}".format(z_ind)].value
            dt = t_1-t_0
            #Ca_close_min.append(1000000)
            #Ca_close_max.append(-1000000)
            for kd_ind, kd_open in enumerate(kd_open_range):
                args.kd_open = kd_open
                p_still_closed_after_t = 1.

                ind_open_ryr = discrete_ryrs.value.nonzero()[0][0]
                x, y = ryrs_inds[ind_open_ryr]
                k_0 = close_rate(data_0[x,y])
                k_1 = close_rate(data_1[x,y])
                p_something_happen[kd_ind][ind_open_ryr] += dt*(k_0*math.exp(-k_0*t_0)+\
                                                                k_1*math.exp(-k_1*t_1))/2
                p_still_open_after_t = 1-p_something_happen[kd_ind][ind_open_ryr]
                
                for ryr_ind, (x, y) in enumerate(ryrs_inds):

                    open_ryr = discrete_ryrs[ryr_ind]
                    if open_ryr:
                        continue
                    
                    # Compute center value of Ca
                    Ca_l_0 = data_0[x,y]
                    Ca_l_1 = data_1[x,y]

                    rate = close_rate if open_ryr else open_rate
                    
                    k_0 = rate(Ca_l_0)
                    k_1 = rate(Ca_l_1)

                    p_something_happen[kd_ind][ryr_ind] += dt*p_still_open_after_t*\
                                                           (k_0*math.exp(-k_0*t_0)+\
                                                            k_1*math.exp(-k_1*t_1))/2
                    p_still_closed_after_t *= 1-p_something_happen[kd_ind][ryr_ind]
                                              
                    
                P_spark[kd_ind].append(1.-p_still_closed_after_t)
                
        for kd_ind, kd_open in enumerate(kd_open_range):
            plt.plot(time_data, P_spark[kd_ind], label="Kd = {}{}".format(\
                kd_open, center_str), lw=2, ls=ls, c=colors[kd_ind])
        
        if center:
            plt.xlabel("time [ms]")
            plt.ylabel(r"$\mathrm{P_s}$ [1]")
            plt.title(r"\textbf{{\#RyRs: {}}}".format(N_ryrs))
            plt.ylim(0,1)

        if first:
            plt.legend(fontsize="xx-small")
        #plt.savefig("figures/{}_prob_spark.pdf".format(args.ifile.replace(".h5", "")))
        
if __name__ == "__main__":
    
    input_files = glob.glob("casename-SRxu-h-12-56-28-*-1-0-cleft-kdo_100_kdc_62_i_0.5-stoch.h5")
    input_files.sort()
    base_scale = 6
    
    args = args().parse_args()
    file_id = 0
    plt_id = 1
    row_num = 3
    col_num = 3
    figsize = [base_scale*row_num, base_scale*col_num]
    plt.rcParams.update({'figure.figsize': figsize})
    fig = plt.figure()
    print input_files, len(input_files)
    for i in range(row_num):
        for j in range(col_num):
            if file_id>=len(input_files):
                continue
            plt.subplot(row_num, col_num, plt_id)
            args.ifile = input_files[file_id]
            plot_rates(args, center=0, first=file_id==0)
            plot_rates(args, first=file_id==0)
            
            file_id += 1
            plt_id += 1

    plt.subplots_adjust(top=0.97, left=0.04, right=0.97, bottom=0.05, hspace=0.30)
    plt.savefig("figures/cum_spark_probability.pdf")
    #plt.show()
