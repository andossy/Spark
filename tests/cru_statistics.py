#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import pylab
import matplotlib.pyplot as plt

def compute_clusters(N, data, neighbors, RyR_cutoff=3):
    # Calculate neighbors
    neighbor_indices = np.array([[set() for j in range(N)] for i in range(N)])
    neighbor_indices.shape =(N*N,)
    
    inds = {
        0  : slice(0,N),
        1  : slice(1,N),
        -1 : slice(0,N-1),
        2  : slice(2,N),
        -2 : slice(0,N-2),
        }

    neigh = np.zeros((N, N), dtype=bool)
    
    for neigh_i, neigh_j in neighbors:

        # Reset neighborinfo
        neigh[:] = 0

        # Find neighbors
        neigh[inds[neigh_i*-1], inds[neigh_j*-1]] = data[inds[neigh_i], inds[neigh_j]]

        # Only include actuall data points with neighbors
        neigh *= data 
        
        # Get global indices of data points with neigbors to its right
        for neig_ind in neigh.flatten().nonzero()[0]:
            neighbor_indices[neig_ind].add(neig_ind+neigh_i*N+neigh_j)

    print "Sort neigboring data. Will take some time..."
    clusters_0 = set([1])
    clusters_1 = set()
    while len(clusters_0) != len(clusters_1):
        clusters_0 = set()

        # Let the neighborhood be expanded with its members
        for my_index in neighbor_indices.nonzero()[0]:

            my_neighbors = set()

            # update myself (neigh_inds) with my neighbors neighbors
            for neigh_ind in neighbor_indices[my_index]:
                my_neighbors.update(neighbor_indices[neigh_ind])

            neighbor_indices[my_index].update(my_neighbors)
            
            # Add my_index to clusters_0
            clusters_0.add(tuple(sorted(neighbor_indices[my_index])))

        tmp = clusters_0
        clusters_0 = clusters_1
        clusters_1 = tmp

    print "Prune small clusters"
    # Flatten data
    data.shape = (N*N,)

    clusters = []
    centers = []
    sizes = []
    num_small_clusters = sum(len(cluster) < RyR_cutoff for cluster in clusters_1)
    for cluster in clusters_1:
        # Remove small clusters
        if len(cluster) < RyR_cutoff and num_small_clusters < len(clusters_1):
            for RyR_ind in cluster:
                data[RyR_ind] = 0
        else:
            # Create RyR lattice coordinates of the indices
            cluster = np.array([[RyR_ind/N, RyR_ind % N] for RyR_ind in cluster], \
                               dtype=float)
            clusters.append(cluster)
            centers.append(np.mean(cluster, 0))
            distances = np.sqrt(np.sum((cluster-centers[-1])**2, 0))
            sizes.append([np.max(distances)*30, np.std(distances)*30])

    print "Calculate size of, and distances between clusters"
    # Numpyfy data
    centers = np.array(centers)
    sizes = np.array(sizes)
    
    # Calculate distances between centers
    X0, X1 = np.meshgrid(centers[:,0], centers[:,0])
    Y0, Y1 = np.meshgrid(centers[:,1], centers[:,1])
    
    dX = X0 - X1
    dY = Y0 - Y1

    dS = np.sqrt(dX**2+dY**2)
    dS[np.diag_indices(len(clusters))] = 1e10
    
    # Get the smallest distance between centers
    cluster_distances = []
    for ind in range(len(clusters)):
        ind_other = dS[ind,:].argmin()
        RyR_distances = np.zeros((len(clusters[ind]), len(clusters[ind_other])))
        for row, RyR_coord_me in enumerate(clusters[ind]):
            for col, RyR_coord_other in enumerate(clusters[ind_other]):
                RyR_distances[row, col] = math.sqrt(np.sum((RyR_coord_me-RyR_coord_other)**2))

        cluster_distances.append([(ind, ind_other), 30*dS[ind,:].min(), \
                                  30*RyR_distances.min()])
        
        
    print "Done"
    # Reshape neighbors back
    neighbor_indices.shape = (N, N)

    # Expand data
    data.shape = (N,N)

    return data, neighbor_indices, clusters, centers, sizes, cluster_distances

def output(clusters, cluster_distances, sizes, nametag=""):
    max_num = 0
    cluster_size = []
    for ind, cluster in enumerate(clusters):
        #print ind, cluster
        max_num = max(max_num, len(cluster))
        cluster_size.append(len(cluster))

    center_distances = []
    closest_distances = []

    for inds, center_dist, closest_dist in cluster_distances:
        center_distances.append(center_dist)
        closest_distances.append(closest_dist)

    print "Mean size:  {:.1f} \xc2\xb1 {:.1f} [{:.1f}, {:.1f}]".format(np.mean(cluster_size), np.std(cluster_size), np.min(cluster_size), np.max(cluster_size))
    print "Mean closest distance: {:.1f} \xc2\xb1 {:.1f} [{:.1f}, {:.1f}]".format(np.mean(closest_distances), np.std(closest_distances), np.min(closest_distances), np.max(closest_distances))
    print "Mean cluster radius: {:.1f} \xc2\xb1 {:.1f} [{:.1f}, {:.1f}]".format(np.mean(sizes[:,0]), np.std(sizes[:,0]), np.min(sizes[:,0]), np.max(sizes[:,0]))
    print "Number Cluster:", len(clusters)
    print "Number RyR:", sum(map(len, clusters))

    pylab.subplots_adjust(left=0.06, bottom=0.05, right=0.97, top=0.97, wspace=0.11)
    pylab.savefig("RyRs_CSQN_distr{0}.png".format(nametag))

    #cluster_size.sort()
    pylab.figure(figsize=(18,6))

    pylab.subplot(131)
    pylab.hist(cluster_size, max_num, normed=False)
    pylab.title("Cluster size")
    pylab.xlabel("Number of RyRs")
    pylab.ylabel("frequency")
    pylab.text(0.5,0.75, "{:.1f} $\\pm${:.1f} [{:.0f},{:.0f}] ".format(
        np.mean(cluster_size),
        np.std(cluster_size),
        np.min(cluster_size),
        np.max(cluster_size),
        ), \
               size="large",
               transform = pylab.gca().transAxes)

    pylab.subplot(132)
    #pylab.hist(center_distances)
    pylab.hist(closest_distances, 20, normed=False)
    pylab.title("Nearest cluster distance (RyR-RyR)")
    pylab.xlabel("Distance [nm]")
    pylab.ylabel("frequency")
    pylab.text(0.5,0.75, "{:.0f} $\\pm${:.0f} [{:.0f},{:.0f}] nm".format(
        np.mean(closest_distances),
        np.std(closest_distances),
        np.min(closest_distances),
        np.max(closest_distances),
        ), \
               size="large",
               transform = pylab.gca().transAxes)

    pylab.subplot(133)
    pylab.hist(sizes[:,0], 20, normed=False)
    pylab.title("Max cluster radius")
    pylab.xlabel("Diameter [nm]")
    pylab.ylabel("frequency")
    pylab.subplots_adjust(left=0.04, bottom=0.1, right=0.98, top=0.95, wspace=0.20)
    pylab.text(0.5,0.75, "{:.0f} $\\pm${:.0f} [{:.0f},{:.0f}] nm".format(
        np.mean(sizes[:,0]),
        np.std(sizes[:,0]),
        np.min(sizes[:,0]),
        np.max(sizes[:,0]),
        ),\
               size="large",
               transform = pylab.gca().transAxes)
    pylab.savefig("Cluster_stat{0}.png".format(nametag))


if __name__ == "__main__":

    data = plt.imread('figures/scale-tresh-binary.tif') # crop2.tif
    data = np.asarray(data[:,:]==0, dtype=bool)
    x_ind_0 = 200
    x_ind_1 = 601
    y_ind_0 = 200
    y_ind_1 = 601
    data = data[x_ind_0:x_ind_1, y_ind_0:y_ind_1].copy()
    N = data.shape[0]
    size_pixel = 0.03
    print data.shape
    RyR_cutoff = 5

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

    # Compute the CRU clusters
    data, neighbor_indices, clusters, centers, sizes, cluster_distances = \
          compute_clusters(N, data, neighbors, RyR_cutoff)

    plt.imshow(data, cmap='Greys', interpolation="nearest", origin="lower", \
               extent=[x_ind_0*size_pixel,x_ind_1*size_pixel,
                       y_ind_0*size_pixel,y_ind_1*size_pixel])
    
    # Output statistics
    output(clusters, cluster_distances, sizes)
    plt.show()
    
    
