import numpy as np
from generate_geometry_file2 import generate_cru_geometry_file

geom_sizes = [(9,9), (14,14), (28,28), (28,28), (56,28), (56,28), (56,56), (56,56), (56,56), (56,56)]
N_RyRs_per_cluster = [1,1,1,5,1,5,1,5,5,5]
N_clusters = [1,1,1,1,1,1,1,1,5,9]

geom_sizes = [(56,28), (56,56), (112,56), (112,112)]
N_RyRs_per_cluster = [21,21,21,21]
N_clusters = [1,1,1,1]

geom_sizes = [(56,28), (56,28), (56,28)]
N_RyRs_per_cluster = [5, 9, 13]
N_RyRs_per_cluster = [5, 9, 13, 21, 29, 37, 45, 57]
N_RyRs_per_cluster = [5]
N_clusters = [1]*len(N_RyRs_per_cluster)
geom_sizes = [(56,56)]*len(N_RyRs_per_cluster)
geom_sizes = [(14,14)]*len(N_RyRs_per_cluster)
def generate_test_geometries(geom_sizes, N_RyRs_per_cluster, N_clusters):
    for (xy, z), N_RyRs, N_cl in zip(geom_sizes, N_RyRs_per_cluster, N_clusters):
        RyR_data = np.zeros((xy,xy), dtype=np.bool)

        prune = []
        
        mids = [(xy/2, xy/2)]
        if N_cl >= 5: 
            mids.extend([(xy/4, xy/4), (3*xy/4, xy/4), (xy/4, 3*xy/4), (3*xy/4, 3*xy/4)])
            
        if N_cl >= 9:
            mids.extend([(xy/7, xy/2), (6*xy/7, xy/2), (xy/2, xy/7), (xy/2, 6*xy/7)])

        if N_cl >= 13:
            mids.extend([(3*xy/8, 3*xy/8), (3*xy/8, 5*xy/8), (5*xy/8, 3*xy/8), (5*xy/8, 5*xy/8)])
            
        for mid_x, mid_y in mids:
           RyR_data[mid_x, mid_y] = True
           if N_RyRs >= 5:
               RyR_data[mid_x+1, mid_y] = True
               RyR_data[mid_x-1, mid_y] = True
               RyR_data[mid_x, mid_y+1] = True
               RyR_data[mid_x, mid_y-1] = True
               
           if N_RyRs >= 9:
               RyR_data[mid_x+1, mid_y+1] = True
               RyR_data[mid_x-1, mid_y-1] = True
               RyR_data[mid_x-1, mid_y+1] = True
               RyR_data[mid_x+1, mid_y-1] = True

           if N_RyRs >= 13:
               RyR_data[mid_x+2, mid_y] = True
               RyR_data[mid_x-2, mid_y] = True
               RyR_data[mid_x, mid_y+2] = True
               RyR_data[mid_x, mid_y-2] = True

           if N_RyRs >= 21:
               RyR_data[mid_x+1, mid_y+2] = True
               RyR_data[mid_x+2, mid_y+1] = True
               RyR_data[mid_x+1, mid_y-2] = True
               RyR_data[mid_x+2, mid_y-1] = True

               RyR_data[mid_x-1, mid_y+2] = True
               RyR_data[mid_x-2, mid_y+1] = True
               RyR_data[mid_x-1, mid_y-2] = True
               RyR_data[mid_x-2, mid_y-1] = True

           if N_RyRs >= 25:
               RyR_data[mid_x+2, mid_y+2] = True
               RyR_data[mid_x+2, mid_y-2] = True
               RyR_data[mid_x-2, mid_y+2] = True
               RyR_data[mid_x-2, mid_y-2] = True

           if N_RyRs >= 29:
               RyR_data[mid_x, mid_y+3] = True
               RyR_data[mid_x, mid_y-3] = True
               RyR_data[mid_x+3, mid_y] = True
               RyR_data[mid_x-3, mid_y] = True

           if N_RyRs >= 37:
               RyR_data[mid_x+1, mid_y+3] = True
               RyR_data[mid_x-1, mid_y+3] = True

               RyR_data[mid_x+1, mid_y-3] = True
               RyR_data[mid_x-1, mid_y-3] = True

               RyR_data[mid_x+3, mid_y+1] = True
               RyR_data[mid_x+3, mid_y-1] = True

               RyR_data[mid_x-3, mid_y+1] = True
               RyR_data[mid_x-3, mid_y-1] = True

           if N_RyRs >= 45:
               RyR_data[mid_x+2, mid_y+3] = True
               RyR_data[mid_x-2, mid_y+3] = True

               RyR_data[mid_x+2, mid_y-3] = True
               RyR_data[mid_x-2, mid_y-3] = True

               RyR_data[mid_x+3, mid_y+2] = True
               RyR_data[mid_x+3, mid_y-2] = True

               RyR_data[mid_x-3, mid_y+2] = True
               RyR_data[mid_x-3, mid_y-2] = True

           if N_RyRs >= 57:
               RyR_data[mid_x, mid_y+4] = True
               RyR_data[mid_x+1, mid_y+4] = True
               RyR_data[mid_x-1, mid_y+4] = True

               RyR_data[mid_x, mid_y-4] = True
               RyR_data[mid_x+1, mid_y-4] = True
               RyR_data[mid_x-1, mid_y-4] = True

               RyR_data[mid_x+4, mid_y] = True
               RyR_data[mid_x+4, mid_y+1] = True
               RyR_data[mid_x+4, mid_y-1] = True

               RyR_data[mid_x-4, mid_y] = True
               RyR_data[mid_x-4, mid_y+1] = True
               RyR_data[mid_x-4, mid_y-1] = True
            
           #if N_RyRs == 57:
           #    prune.extend([3,7,8,9,10,12,13,14,18,19,20,28,29,37,38,42,\
           #                  43,44,45,46,47,48,49,53])

        generate_cru_geometry_file(\
            RyR_data, Nz=z, basename="test_geometry_SR_xy_{}_z_{}_N_RyR_{}".format(\
                xy, z, (N_RyRs-len(prune))*N_cl),
            save2d_pdf=True, prune_RyRs=prune)

if __name__ == "__main__":
    generate_test_geometries(geom_sizes, N_RyRs_per_cluster, N_clusters)
