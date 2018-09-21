import sys
import matplotlib.pyplot as plt
import time
import h5py

directions = dict(x="yz", y="xz", z="xy")
extent_inds = dict(x=[1,2], y=[0,2], z=[0,1])

def plot_sheets(casename):

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
        
        Ca_in_data = "Ca" in species_names
        species_names.append("Ca")
        species_names.sort()
            
        base_scale = 6
        figsize = [len(species_names)*base_scale, base_scale*len(indices)]
        plt.rcParams.update({'figure.figsize': figsize})
        #plt.interactive(True)
        fig = plt.figure()

        n_plots = len(f.keys())
        imshow_objs = []
        plt.clf()
        for data_ind, data_name in enumerate(f.keys()):
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
            plot_ind = 1
            t = f[data_name].attrs["time"]
            print "time:", t
            for ind in indices:
                Ca_first = 0
                for species in species_names:
                    print "reading:", "{}_{}_sheet_{}".format(\
                        species, sheet_plane, ind)
                    data = f[data_name]["{}_{}_sheet_{}".format(\
                        species, sheet_plane, ind)].value

                    mM = data.max() > 1000

                    if mM:
                        data /= 1000
                        unit = "mM"
                    else:
                        unit = "uM"

                    # First time we collect minimal fluo value and use to scale the plot
                    if "Flou" in species and data_ind == 0:
                        fluo_0 = data.min()

                    if "Flou" in species:
                        data -= fluo_0
                        data /= fluo_0
                        data += 1.0
                        unit = "F/F0"
                    
                    plt.subplot(len(indices)*100 + len(species_names)*10 + plot_ind)
                    if data_ind == 0:
                        imshow_objs.append(plt.imshow(data, extent=extent, origin="lower"))
                        if species == "Ca":
                            if not Ca_first:
                                Ca_first = 1
                                plt.clim(0., 1.0)
                            else:
                                plt.clim(0., 200.0)
                                
                        elif "Flou" in species:
                            plt.clim(1., 4.0)
                        
                        plt.colorbar()
                        
                    else:
                        imshow_objs[plot_ind-1].set_data(data)

                    plot_ind += 1
                    plt.title(r"$\mathrm{{{}_{{{}}}[{}]:t={:.2f}ms:{}/{}}}$".format(\
                        species, sheet_direction, unit, t[0], data_ind, n_plots))
                    plt.xlabel("[um]")
                    plt.ylabel("[um]")

            
            plt.draw()
            plt.savefig("figures/movies/" + casename.replace(\
                ".h5", "_{:04d}.png".format(data_ind)))
            #time.sleep(.002)

        #plt.interactive(True)
        #plt.show()
        #time.sleep(10000.0)
            
if __name__ == "__main__":
    
    casename = sys.argv[-1] if len(sys.argv)>=1 else "casename"
    plot_sheets(casename)
    #plt.show()
