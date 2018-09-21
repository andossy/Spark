import pylab
import glob
import re
import numpy as np

#pylab.rcParams['ps.useafm'] = True
#pylab.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
#pylab.rcParams['pdf.fonttype'] = 42

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
species_output = ["Cai", "CaSR", "CaFluo"]#, "CaCSQN"]
title_form = dict(Cai="$c$", CaSR="$c^{\mathrm{sr}}$", CaCMDN="$c^{\mathrm{sr}}$", CaATP="$c^{\mathrm{ATP}}$", CaFluo="$c^{\mathrm{Fluo4}}$", CaTRPN="$c^{\mathrm{TRPN}}$", CaCSQN="$c^{\mathrm{CSQN}}$")

zoom_extent = [0.8,1.3,3,3.5]
zoom_distance = [[zoom_extent[2]*1000,zoom_extent[3]*1000],
                 [zoom_extent[0]*1000,zoom_extent[1]*1000]]

time_serie = [7.]

base_scale = 5
landscape = False
if landscape:
    figsize = [len(species_output)*base_scale, 2*base_scale]
else:
    figsize = [2*base_scale, len(species_output)*base_scale]
pylab.rcParams.update({'figure.figsize': figsize})
#pylab.interactive(True)
#figure = pylab.figure(figsize=figsize)

# First time we collect minimal fluo value and use to scale the plot
if "CaFluo" in species_output:
    print "Load Fluo data"
    data = pylab.loadtxt("merge_dir_%d_%d_%d_%d_%d/Ca%d_T%d_merge.np" % \
                     (h, nx, ny, nz, use_failing, 4, 0))
    Ca_fluo_0 = data.min()
    all_csqn_files = glob.glob("merge_dir_%d_%d_%d_%d_%d/Ca%d_T*_merge.np" % \
                               (h, nx, ny, nz, use_failing, 6))

if "CaCSQN" in species_output:
    print "Load CSQN data"
    re_pattern = "merge_dir_%d_%d_%d_%d_%d/Ca%d_T([0-9]+)_merge.np" % \
                 (h, nx, ny, nz, use_failing, 6)

    last_csqn_ind = sorted(int(re.findall(re_pattern, f)[0]) for f in all_csqn_files)[-1]
    print last_csqn_ind

    last_csqn_data = pylab.loadtxt("merge_dir_%d_%d_%d_%d_%d/Ca%d_T%d_merge.np" % \
                                   (h, nx, ny, nz, use_failing, 6, last_csqn_ind))

    max_csqn = last_csqn_data.max()-0.00001
    ind_no_csqn = last_csqn_data >= max_csqn

pylab.clf()
if landscape:
    pylab.subplots_adjust(left=0.035, right=0.99, bottom=0.035, top=0.98)
else:
    pylab.subplots_adjust(left=0.05, right=0.99, bottom=0.03, top=0.99)

for t in time_serie:
    TT = int(t/DT)
    print "frame", TT
    for local_ind, species in enumerate(species_output):
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
            unit = "$\mu$M"

        for zoom in [0,1]:
            
            if species == "CaFluo" and zoom == 0:
                data -= Ca_fluo_0
                data /= Ca_fluo_0
                data += 1.0
                unit = "F/F0"
            
            if species == "CaCSQN":
                data[ind_no_csqn] = species_limits[local_ind][0]
                data_mean = data[ind_no_csqn==0].mean()
                data_max = data[ind_no_csqn==0].max()
                data_min = data[ind_no_csqn==0].min()
            else:
                data_mean = data.mean()
                data_max = data.max()
                data_min = data.min()
            
            if landscape:
                ax = pylab.subplot(2, len(species_output), local_ind + zoom*len(species_output))
            else:
                ax = pylab.subplot(len(species_output), 2, local_ind*2 + zoom + 1)

            rect = ax.get_frame()
            dpi = rect.figure.get_dpi()
            fig_width = rect.figure.get_figwidth()*dpi
            fig_height = rect.figure.get_figheight()*dpi

            

            left, right, bottom, top = rect.get_extents().bounds
            width = (right - left)/fig_width
            height = (bottom - top)/fig_height
            x0 = left/fig_width
            y0 = bottom/fig_height

            if not zoom:
                pylab.imshow(data[1:-1,1:-1], extent=extent, origin="lower")
                pylab.clim(species_limits[species_ind])
                pylab.text(-0.1, -0.22, 'Limits %s: [%.2f, %.1f] %s; mean %.2f %s' % \
                           (title_form[species], data_min, data_max, unit, data_mean, unit),
                           horizontalalignment='left', \
                           transform = pylab.gca().transAxes)
                xy = (zoom_extent[0], zoom_extent[2])
                width = zoom_extent[1] - zoom_extent[0]
                height = zoom_extent[3] - zoom_extent[2]
                ax.add_patch(pylab.Rectangle(xy, width, height, fill=False,
                                             ls="dotted"))
                #print "Add Rectangle:", xy, width, height
                
            else:
                m_range = np.arange(int(zoom_distance[0][0]/3), int(zoom_distance[0][1]/3))
                n_range = np.arange(int(zoom_distance[1][0]/3), int(zoom_distance[1][1]/3))
                pylab.imshow(data[:, n_range][m_range,:], extent=zoom_extent, origin="lower")
                if species == "Cai":
                    limits = species_limits[species_ind]
                    pylab.clim([limits[0], limits[1]*5])


            pylab.colorbar()
            pylab.xlabel("[$\mu$m]")
            pylab.ylabel("[$\mu$m]")
            pylab.title("%s at t=%.0f ms" % (title_form[species], TT*DT))

        pylab.draw()

if landscape:
    outfigname="R%dnm_%d_%d_%d_full_zoom"%(h, nx, ny, nz)
else:
    outfigname="R%dnm_%d_%d_%d_full_zoom_portrait"%(h, nx, ny, nz)

pylab.savefig(outfigname+".png")
pylab.savefig(outfigname+".pdf")

#pylab.show()
