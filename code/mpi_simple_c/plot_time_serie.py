import pylab
import glob
import re

pylab.rcParams['ps.useafm'] = True
pylab.rc('font',**{'family':'sans-serif','sans-serif':['FreeSans']})
pylab.rcParams['pdf.fonttype'] = 42

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
species_output = ["Cai", "CaSR", "CaCSQN"]
time_serie = [1., 8, 24]

title_form = dict(Cai="$c$", CaSR="$c^{\mathrm{sr}}$", CaCMDN="$c^{\mathrm{sr}}$", CaATP="$c^{\mathrm{ATP}}$", CaFluo="$c^{\mathrm{Fluo4}}$", CaTRPN="$c^{\mathrm{TRPN}}$", CaCSQN="$c^{\mathrm{CSQN}}$")

base_scale = 5
landscape = False
if landscape:
    figsize = [len(species_output)*base_scale, len(time_serie)*base_scale]
else:
    figsize = [len(time_serie)*base_scale, len(species_output)*base_scale]

pylab.rcParams.update({'figure.figsize': figsize})
#pylab.interactive(True)
#figure = pylab.figure(figsize=figsize)

# First time we collect minimal fluo value and use to scale the plot
data = pylab.loadtxt("merge_dir_%d_%d_%d_%d_%d/Ca%d_T%d_merge.np" % \
                     (h, nx, ny, nz, use_failing, 4, 0))
Ca_fluo_0 = data.min()
all_csqn_files = glob.glob("merge_dir_%d_%d_%d_%d_%d/Ca%d_T*_merge.np" % \
                                  (h, nx, ny, nz, use_failing, 6))

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
    pylab.subplots_adjust(left=0.05, right=0.99, bottom=0.02, top=0.99)
for time_ind, t in enumerate(time_serie):
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

        if species == "CaFluo":
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
            pylab.subplot(len(time_serie), len(species_output), \
                          time_ind*len(species_output) + local_ind+1)
        else:
            pylab.subplot(len(species_output), len(time_serie),\
                          local_ind*len(time_serie) + time_ind+1)
            
        pylab.imshow(data[1:-1,1:-1], extent=extent, origin="lower")
        pylab.clim(species_limits[species_ind])

        pylab.colorbar()
        pylab.xlabel("[$\mu$m]")
        pylab.ylabel("[$\mu$m]")
        pylab.text(-0.1, -0.22, 'Limits %s: [%.2f, %.1f] %s; mean %.2f %s' % \
                   (title_form[species], data_min, data_max, unit, data_mean, unit),
                   horizontalalignment='left', \
                   transform = pylab.gca().transAxes)
        pylab.title("%s at t=%.0f ms" % (title_form[species], TT*DT))

        pylab.draw()

if landscape:
    outfigname="R%dnm_%d_%d_%d_time_series"%(h, nx, ny, nz)
else:
    outfigname="R%dnm_%d_%d_%d_time_series_portrait"%(h, nx, ny, nz)
    
pylab.savefig(outfigname+".pdf")
pylab.savefig(outfigname+".png")

#pylab.show()
