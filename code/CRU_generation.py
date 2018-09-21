#!/usr/bin/python
# -*- coding: utf-8 -*-
import math
import numpy as np
import pylab

fig_size =  [12,6]
params = {'figure.figsize': fig_size}
pylab.rcParams.update(params)
pylab.interactive(True)

global enable_output

enable_output = False

def CRU_generation(jada):
    pass

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
        
        
    # Reshape neighbors back
    neighbor_indices.shape = (N, N)

    # Expand data
    data.shape = (N,N)

    return data, neighbor_indices, clusters, centers, sizes, cluster_distances

def growth_model(iterations=100,
                 N = 200,
                 P_nucleation=2.5e-4,
                 P_growth=2.5e-2,
                 P_retention=0.937,
                 RyR_cutoff=3,
                 neighbors=[[1,0], [-1,0], [0,1], [0,-1]]):
    
    shape = (N, N)
    data = np.zeros(shape, dtype=np.bool)
    csqn_data = np.zeros(shape, dtype=np.bool)

    P_mat = np.asarray(data, dtype=float)
    indm = np.arange(0,N-1)
    indp = np.arange(1,N)
    
    if enable_output:
        pylab.figure()

    indp = slice(1,N)
    indm = slice(0,N-1)
    
    for it in range(iterations):
        # Reset data
        P_mat[:] = 0.0

        # Compute neighbors in xy direction
        P_mat[indm, :] = data[indp, :]
        P_mat[indp, :] += data[indm, :]
        P_mat[:, indm] += data[:, indp]
        P_mat[:, indp] += data[:, indm]

        # Compute neighbors in xy diagonals
        P_mat[indm, indm] += data[indp, indp]
        P_mat[indp, indp] += data[indm, indm]
        P_mat[indp, indm] += data[indm, indp]
        P_mat[indm, indp] += data[indp, indm]

        P_mat *= P_growth

        # Set prob for staying active
        P_mat[data] = P_retention

        if it % 10 == 0 and enable_output:
            csqn_data[:] = 0
            csqn_data[:] = P_mat
            
            pylab.subplot(121)
            pylab.imshow(data, interpolation="nearest", origin="lower", \
                         extent=[0,N*0.030,0,N*0.030])
            pylab.subplot(122)
            pylab.imshow(csqn_data, interpolation="nearest", origin="lower",\
                         extent=[0,N*0.030,0,N*0.030])
            pylab.draw()
            print it, iterations
            
        # Set prob for generate new RyR
        P_mat[P_mat==0.0] = P_nucleation
        
        # Update RyR data
        data[:] = np.random.random(shape) < P_mat

        P_mat[:] = 0.0

    # If empty add fake RyR
    if not data.any():
        data[N/2,N/2] = True

    data, neighbor_indices, clusters, centers, sizes, cluster_distances = \
          compute_clusters(N, data, neighbors, RyR_cutoff)

    # Compute neighbors in xy direction
    csqn_data[:] = 0
    csqn_data[indm, :] = data[indp, :]
    csqn_data[indp, :] += data[indm, :]
    csqn_data[:, indm] += data[:, indp]
    csqn_data[:, indp] += data[:, indm]
    csqn_data[data] = 1

    # Compute neighbors in xy diagonals
    csqn_data[indm, indm] += data[indp, indp]
    csqn_data[indp, indp] += data[indm, indm]
    csqn_data[indp, indm] += data[indm, indp]
    csqn_data[indm, indp] += data[indp, indm]

    if enable_output:
        pylab.subplot(121)
        pylab.imshow(data, interpolation="nearest", origin="lower",\
                     extent=[0,N*0.030,0,N*0.030])
        pylab.title("RyR clusters")
        pylab.xlabel("$\mu$m")
        pylab.ylabel("$\mu$m")
        pylab.subplot(122)
        pylab.imshow(csqn_data, interpolation="nearest", origin="lower",\
                     extent=[0,N*0.030,0,N*0.030])
        pylab.title("CSQN distribution")
        pylab.xlabel("$\mu$m")
        pylab.ylabel("$\mu$m")
        pylab.draw()

    return data, csqn_data, neighbor_indices, clusters, centers, sizes, cluster_distances

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

    # Remove to get other distribution
    #np.random.seed(0)

    enable_output = False
    # Original data
    # iterations=100
    # N=100
    # P_nucleation=2.5e-4,
    # P_growth=2.5e-2,
    # P_retention=0.937):

    iterations=150
    N = 10000/30
    P_nucleation=2.5e-4  # Healthy 0.5e-4
    P_growth=1.4e-2     # Healthy 1.40e-2
    P_retention=0.915   # Healthy 0.937

    RyR_cutoff = 3

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
    data, csqn_data, neighbor_indices, clusters, centers, sizes, cluster_distances = \
          growth_model(iterations, N, P_nucleation, P_growth, P_retention, RyR_cutoff,
                       neighbors)

    # Save index arrays
    #print data.nonzero()
    data.nonzero()[0].tofile(open("i_RyR_indices.csv", "w"), sep=", ")
    data.nonzero()[1].tofile(open("j_RyR_indices.csv", "w"), sep=", ")
    csqn_data.nonzero()[0].tofile(open("i_csqn_indices.csv", "w"), sep=", ")
    csqn_data.nonzero()[1].tofile(open("j_csqn_indices.csv", "w"), sep=", ")
    
    # Output statistics
    output(clusters, cluster_distances, sizes)
    #pylab.show()
    
    
