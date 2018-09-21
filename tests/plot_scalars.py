import argparse
import sys
import matplotlib.pyplot as plt
import time
import h5py

def plot_scalars(ifile, plot_domains, plot_species, stats):

    with h5py.File(ifile) as f:

        h = f.attrs["h"]
        N = f.attrs["N"]
        if "data_0000000" not in f.keys():
            raise ValueError("No data in file")

        # Read domains
        with h5py.File(f.attrs["geometry_file"]) as fg:
            domains = fg["domains"].attrs
            num_domains = domains["num"]
            domains = [domains["name_{}".format(ind)] for ind in range(num_domains)]

        if plot_domains is not None:
            for dom in plot_domains:
                if dom not in domains:
                    raise ValueError("'{}' is not a domain in '{}' [{}]".format(dom, ifile, ", ".join(domains)))
        else:
            plot_domains = domains

        # Read species
        with h5py.File(f.attrs["species_file"]) as fs:
            num_species = fs.attrs["num_species"]
            species = [fs.attrs["species_name_{}".format(ind)] for ind in range(num_species)]

        if plot_species is not None:
            for sp in plot_species:
                if sp not in species:
                    raise ValueError("'{}' is not a species in '{}' [{}]".format(dom, ifile, ", ".join(species)))
        else:
            plot_species = species

        times = []
        data = dict((domain, dict((species, dict((stat, []) for stat in stats)) for species in plot_species)) for domain in plot_domains)
        
        discrete_ryr_data = []
        for data_ind, data_name in enumerate(f.keys()):
            times.append(f[data_name].attrs["time"])
            discrete_ryr_data.append(f[data_name]["discrete_ryr"].value.sum())
            for domain in plot_domains:
                for species in plot_species:
                    for stat in stats:
                        data[domain][species][stat].append(\
                            f[data_name][domain][species].attrs[stat])
        
        base_scale = 6
        figsize = [base_scale*(len(plot_domains)+args.include_ryr_openings), \
                   len(plot_species)*base_scale]
        #plt.rcParams.update({'figure.figsize': figsize})
        #plt.interactive(True)
        fig = plt.figure(figsize=figsize)

        c = dict(average="-b", min="--b", max="-.b",)
        plot_ind = 1
        subplot_base = len(plot_species)*100 + (len(plot_domains)+\
                                                args.include_ryr_openings)*10
        for ind_s, species in enumerate(plot_species):
            for ind_d, domain in enumerate(plot_domains):
                plt.subplot(subplot_base + plot_ind)
                for stat in stats:
                    plt.plot(times, data[domain][species][stat], c[stat],
                             label="{}".format(stat), lw=2)
                    plt.xlabel("time [ms]")
                    plt.ylabel(r"[{}] $\mu$M".format(species))
                    plt.title(r"$\mathrm{{{}_{{{}}}}}$".format(species, domain))
                plot_ind += 1

            plt.legend()

            # If first species and include_ryr_openings
            if ind_s == 0 and args.include_ryr_openings:
                plt.subplot(subplot_base + plot_ind)
                plt.step(times, discrete_ryr_data, lw=2)
                plt.xlabel("time [ms]")
                plt.ylabel("\# open RyRs")
                
            plot_ind += args.include_ryr_openings
                    
        plt.subplots_adjust(left=0.05, right=0.97, bottom=0.10, top=0.95)

def args():
    descr = "plot scalar spark data from .h5 file."
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("ifile", type=str, help="HDF5 file .h5 filename.")
    parser.add_argument("-d", "--domains", type=str, metavar="cyt", nargs="*",
                        help="domains that should be plotted. If not given all "\
                        "domains will be plotted.")
    parser.add_argument("-S", "--species", type=str, metavar="Ca", nargs="*",
                        help="species that will be plotted. If not give all species "\
                        "will be plotted.")
    parser.add_argument("-s", "--stats", type=str, metavar="min", nargs="*", \
                        default=["max", "average", "min"], choices=["max", "average", "min"],\
                        help="species that will be plotted. If not give all species "\
                        "will be plotted.")
    parser.add_argument("-r", "--include-ryr-openings",
                        dest="include_ryr_openings", action="store_true",
                        help="if true ryr openings will be plotted.")
    return parser
            
if __name__ == "__main__":

    args = args().parse_args()
    plot_scalars(args.ifile, args.domains, args.species, args.stats)
    plt.savefig("figures/" + args.ifile.replace(".h5", ".pdf"))
    plt.show()
