#!/usr/bin/python
# -*- coding: utf-8 -*-
from CRU_generation import growth_model, pylab, output
import numpy as np
np.random.seed(2)

cmap = pylab.get_cmap("binary")

domain_size = 10 # um

iterations=150
#assert((domain_size*1000) % 30 == 0)
N = domain_size*1000/30+1
RyR_cutoff = 3
P_growth=1.4e-2      # Healthy 1.40e-2
num_sarcomeres = 10
# Healthy stats:
# Mean size:  13.9 ± 14.0 [3.0, 85.0]
# Mean closest distance: 222.7 ± 121.6 [90.0, 721.2]
# Mean cluster radius: 242.5 ± 288.3 [24.5, 2174.0]

# Set the relative coordinates of who are neighbors
diagonals = [[1,1], [1,-1], [-1,1], [-1,-1]]
neighbors = [[0,0], [1,0], [-1,0], [0,1], [0,-1]]
neighbors_2nd = [[2,2], [2,1], [2,0], [2,-1], [2,-2], \
                 [-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2],\
                 [0,2], [0,-2], [1,2], [1,-2], [-1,2], [-1,-2]]

# Add all diagonals to neigbors
neighbors += diagonals

# Also add all 2nd neighbors
neighbors += neighbors_2nd

P_vec = [[0.125e-4, 0.96], # Healthy [0.5e-4, 0.937]
         [10.e-4, 0.892]] # Failing [2.5e-4, 0.915]
#P_vec = [[0.5e-4, 0.937],
#         [2.5e-4, 0.915]]

base_size = 6
f=pylab.figure(figsize=(base_size*2, base_size*len(P_vec)))
pylab.interactive(False)

for ind_sarcomere in range(num_sarcomeres):
    pylab.clf()
    for ind_failure, (P_nucleation, P_retention) in enumerate(P_vec):
        
        # Compute the CRU clusters
        data, csqn_data, neighbor_indices, clusters, centers, sizes, cluster_distances = \
              growth_model(iterations, N, P_nucleation, P_growth, P_retention, RyR_cutoff,
                           neighbors)
        if ind_failure==1:
            output(clusters, cluster_distances, sizes, "failing")
        else:
            output(clusters, cluster_distances, sizes)
        pylab.figure(f.number)
        pylab.subplot(len(P_vec), 2 ,1+ind_failure*len(P_vec))
        pylab.imshow(data, interpolation="nearest", origin="lower",\
                     extent=[0,N*0.030,0,N*0.030], cmap=cmap)
        title = "RyR clusters sarc: {}".format(ind_sarcomere)
        if ind_failure:
            title += " (failing)"
        pylab.title(title)
        pylab.xlabel("$\mu$m")
        pylab.ylabel("$\mu$m")
    
        pylab.figure(f.number)
        pylab.subplot(len(P_vec), 2, 2+ind_failure*(len(P_vec)))
        pylab.imshow(csqn_data, interpolation="nearest", origin="lower",\
                     extent=[0,N*0.030,0,N*0.030], cmap=cmap)
        title = "CSQN distribution sarc: {}".format(ind_sarcomere)
        if ind_failure:
            title += " (failing)"
        pylab.title(title)
        pylab.xlabel("$\mu$m")
        pylab.ylabel("$\mu$m")
    
        # Save indices to file
        failing = "" if ind_failure == 0 else "_failing"
        data.nonzero()[0].tofile(open("i_RyR_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
        data.nonzero()[1].tofile(open("j_RyR_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
        csqn_data.nonzero()[0].tofile(open("i_csqn_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
        csqn_data.nonzero()[1].tofile(open("j_csqn_indices_sarc_{}{}.dat".format(ind_sarcomere, failing), "w"), sep="\n")
    
    # Output statistics
    #output(clusters, cluster_distances)
    pylab.subplots_adjust(left=0.04, bottom=0.07, right=0.99, top=0.95, wspace=0.04, hspace=0.13)
    pylab.draw()
    f.savefig("cluster_distribution_sarc_{}.pdf".format(ind_sarcomere))

#pylab.show()
    
    
