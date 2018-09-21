import pylab


try:
    param_file = "param"
    param_lines = open(param_file).read().replace(" ", "").split("\n")
except:
    param_file = "../param"
    param_lines = open(param_file).read().replace(" ", "").split("\n")

for ind in range(0, len(param_lines)-2, 2):
    exec("{0} = {1}({2})".format(\
        param_lines[ind], "float" if param_lines[ind] in ["T", "DT"] else "int",
        param_lines[ind+1]))

nx = size_x
ny = size_y
nz = size_z

extent = [0, ny*h/1000., 0, nz*h/1000.]

print nx, ny, nz
print DT, T
#T = 0
t = 0
counter = 0;
mean = [];
#DT = 0.01; # plotting time step

num_frames = int(T/DT + 1)
#num_frames = 5

species_map = ["Cai", "CaSR", "CaCMDN", "CaATP", "CaFluo", "CaTRPN", "CaCSQN"]
species_limits = [[0.1,1.0], [.5, 1.4], [0,25], [0, 455], [1., 3.], [0, 70], [10, 20]]
species_output = ["Cai", "CaSR", "CaFluo", "CaCSQN"]

base_scale = 6
figsize = [len(species_output)*base_scale, base_scale]
pylab.rcParams.update({'figure.figsize': figsize})
#figure = pylab.figure(figsize=figsize)
pylab.interactive(True)
for TT in range(num_frames):
    print "frame", TT 
    pylab.clf()
    pylab.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    for ind, species in enumerate(species_output):
        species_ind = species_map.index(species)
        print "loading:", "merge_dir_%d_%d_%d_%d_%d/Ca%d_T%d_merge.np" % \
              (h, nx, ny, nz, use_failing, species_ind, TT)
        data = pylab.loadtxt("merge_dir_%d_%d_%d_%d_%d/Ca%d_T%d_merge.np" % \
                             (h, nx, ny, nz, use_failing, species_ind, TT))

        mM = data.max() > 1000

        if mM:
            data /= 1000
            unit = "mM"
        else:
            unit = "uM"

        #if species == "CaCSQN":
        #    data_max = data.max()
        #    ind_data_max = data > data_max-0.1
        #    data[ind_data_max] = species_limits[ind][0]

        # First time we collect minimal fluo value and use to scale the plot
        if species == "CaFluo" and TT == 0:
            Ca_fluo_0 = data.min()

        if species == "CaFluo":
            data -= Ca_fluo_0
            data /= Ca_fluo_0
            data += 1.0
            unit = "F/F0"
        
        pylab.subplot(101+ind+len(species_output)*10)
        pylab.imshow(data[1:-1,1:-1], extent=extent, origin="lower")
        pylab.clim(species_limits[species_ind])

        pylab.colorbar()
        pylab.xlabel("[um]")
        pylab.ylabel("[um]")
        pylab.text(-0.1, -0.18, 'Limits %s: [%.2f, %.1f] %s; mean %.2f %s' % \
                   (species, data.min(), data.max(), unit, data.mean(), unit),
                   horizontalalignment='left', \
                   transform = pylab.gca().transAxes)
        pylab.title("%s at t=%.2f ms" % (species, TT*DT))

    pylab.draw()
    #outfigname="R%dnm_%02dFrame.png"%(h,TT)
    #pylab.savefig(outfigname)

pylab.show()
