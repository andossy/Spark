import sys
import matplotlib.pyplot as plt
import time
import h5py

directions = dict(x="yz", y="xz", z="xy")
extent_inds = dict(x=[1,2], y=[0,2], z=[0,1])

def plot_sheets(casename, data_points):

    print casename, data_points
    fluo_0 = 1.
    print casename
    with h5py.File(casename) as f:
        h = f.attrs["h"]
        N = f.attrs["N"]
        if "data_0000000" not in f.keys():
            raise ValueError("No data in file")
        sheet_names = [data_key for data_key in f["data_0000000"].keys()
                       if "sheet" in data_key]

        print sheet_names
        species_names = sorted(set(name.split("_sheet_")[0] for name in sheet_names))

        # Get sheet direction
        sheet_plane = species_names[0][-1]
        sheet_direction = directions[sheet_plane]
        extent_ind = extent_inds[sheet_plane]
        species_names = [name[:-2] for name in species_names]
        extent = [0, N[extent_ind[0]]*h/1000., 0, N[extent_ind[1]]*h/1000.]
        
        indices = sorted(set(int(name.split("_sheet_")[1]) for name in sheet_names))
        
        if len(sheet_names) == 0:
            raise ValueError("No sheets stored in file.")
        
        #Ca_in_data = "Ca" in species_names
        #species_names.append("Ca")
        #species_names.sort()
            
        base_scale = 6
        figsize = [len(species_names)*base_scale*1.5, base_scale*len(indices)]
        plt.rcParams.update({'figure.figsize': figsize})
        fig = plt.figure()
        plt.clf()
        time_data = []
        species_data = [[[[] for _ in range(len(data_points))] \
                           for __ in range(len(indices))] \
                         for ___ in range(len(species_names))]

        discrete_ryr_data = []
        print species_data
        
        for data_ind, data_name in enumerate(f.keys()):
            t = f[data_name].attrs["time"][-1]
            time_data.append(t)
            discrete_ryr_data.append(f[data_name]["discrete_ryr"].value.sum())
            print "time:", t
            for ind_ind, ind in enumerate(indices):
                Ca_first = 0
                for s_ind, species in enumerate(species_names):
                    print "reading:", "{}_{}_sheet_{}".format(\
                        species, sheet_plane, ind)
                    data = f[data_name]["{}_{}_sheet_{}".format(\
                        species, sheet_plane, ind)].value
                    for xy_ind, (x, y) in enumerate(data_points):
                        species_data[ind_ind][s_ind][xy_ind].append(data[x,y])
                        #print ind_ind, s_ind, xy_ind, [x, y], data[x,y], len(species_data[ind_ind][s_ind][xy_ind])

        plt.subplots_adjust(left=0.10, right=0.95, bottom=0.05, top=0.95)
        plot_ind = 1
        for ind_ind, ind in enumerate(indices):
            for s_ind, species in enumerate(species_names):
                plt.subplot(len(indices)*100 + len(species_names)*10 + plot_ind)
                for xy_ind, (x,y) in enumerate(data_points):
                    print ind_ind, s_ind, xy_ind, [x, y]
                    print len(time_data), len(species_data[ind_ind][s_ind][xy_ind])
                    plt.plot(time_data, species_data[ind_ind][s_ind][xy_ind], lw=2)
                plt.xlabel("time [ms]")
                plt.ylabel(" {} [uM]".format(species))
                plot_ind += 1
        plt.legend([str(p) for p in data_points])
        #plt.interactive(True)
        plt.savefig("figures/{}_scalars_in_sheets.pdf".format(casename.replace(".h5", "")))

        # Plot discrete ryr data
        fig = plt.figure()
        figsize = [base_scale*1.5, base_scale]
        plt.rcParams.update({'figure.figsize': figsize})
        plt.step(time_data, discrete_ryr_data, lw=2)
        plt.xlabel("time [ms]")
        plt.ylabel("\# open RyRs")
        plt.savefig("figures/{}_open_ryrs.pdf".format(casename.replace(".h5", "")))
        plt.show()
        #time.sleep(10000.0)
            
if __name__ == "__main__":
    
    casename = sys.argv[-2] if len(sys.argv)>=2 else \
               (sys.argv[-1] if len(sys.argv)>=1 else "casename" )
    data_points = eval(sys.argv[-1]) if len(sys.argv)>=1 else [(0,0)]
    plot_sheets(casename, data_points)
    #plt.show()
