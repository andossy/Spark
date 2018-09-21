from generate_cru_geometry_file import generate_cru_geometry_file
import numpy as np
import geometry_util as gu
import logging

logging.getLogger().setLevel(0)

def U1_G12359():
    nx = ny = 28 # 1.008 microns
    padding = 1
    same_jsr = True
    subdivide = 3 # 36 microns to 12 microns

    ryr = {n: np.zeros((nx, ny)) for n in (1, 2, 3, 5, 9)}

    ryr[1][nx/2, ny/2] = 1
    ryr[2][nx/2-1:nx/2+1, ny/2] = 1
    ryr[3][nx/2-1:nx/2+2, ny/2] = 1
    ryr[5][nx/2-1:nx/2+2, ny/2] = 1; ryr[5][nx/2, ny/2-1:ny/2+2] = 1
    ryr[9][nx/2-1:nx/2+2, ny/2-1:ny/2+2] = 1

    if same_jsr == True:
        jsr = {n: gu.create_jsr(ryr[9], padding) for n in (1, 2, 3, 5, 9)}
    else:
        jsr = {n: gu.create_jsr(ryr[n], padding) for n in (1, 2, 3, 5, 9)}

    # Refine RyR and jSR
    for n in (1, 2, 3, 5, 9):
        ryr[n] = gu.refine_ryr(ryr[n], subdivide)
        jsr[n] = gu.refine_jsr(jsr[n], subdivide, remove_edges=True)

    assert all([np.sum(ryr[n]) == n for n in (1, 2, 3, 5, 9)])

    nsr = gu.create_nsr_from_ryr(ryr[9])
    ttub = jsr.copy()

    # for r, ryr in enumerate((ryr1, ryr2, ryr3, ryr5, ryr9)):
    #     generate_cru_geometry_file(ryr, jsr, nsr, tt=ttub, nz=nx*3,
    #                                nsr_height=36, sr_volume=None,
    #                                cleft_size=1, jsr_thickness=3,
    #                                double=True, basename="geometries/idealized/U{}_R{}".format(p, [1,2,3,5,9][r]))


def U1_G59121625():
    nx = ny = 28 # 1.008 microns
    padding = 1
    same_jsr = True
    subdivide = 3 # 36 microns to 12 microns
    cases = (4, 6, 8, 12, 16, 20, 24)

    ryr = {n: np.zeros((nx, ny)) for n in cases}

    cx = nx/2
    cy = ny/2

    ryr[4][cx-1:cx+1, cy-1:cy+1] = 1

    ryr[6][cx-2:cx+1, cy-1:cy+1] = 1
    
    ryr[8][cx-2:cx+2, cy-1:cy+1] = 1
    
    ryr[12] += ryr[8]
    ryr[12][cx-1:cx+1, cy-2] = 1
    ryr[12][cx-1:cx+1, cy+1] = 1
    
    ryr[16][cx-2:cx+2, cy-2:cy+2] = 1
    
    ryr[20] += ryr[16]
    ryr[20][cx-3, cy-1:cy+1] = 1
    ryr[20][cx+2, cy-1:cy+1] = 1
    
    ryr[24] += ryr[20]
    ryr[24][cx-1:cx+1, cy-3] = 1
    ryr[24][cx-1:cx+1, cy+2] = 1
    
    if same_jsr == True:
        jsr = {n: gu.create_jsr(ryr[cases[-1]], padding) for n in cases}
    else:
        jsr = {n: gu.create_jsr(ryr[n], padding) for n in cases}

    # Refine RyR and jSR
    for n in cases:
        ryr[n] = gu.refine_ryr(ryr[n], subdivide)
        jsr[n] = gu.refine_jsr(jsr[n], subdivide, remove_edges=True)


    nsr = {n: gu.create_nsr_from_ryr(ryr[cases[-1]]) for n in cases}
    ttub = {n: gu.create_ttub_from_ryr(ryr[cases[-1]]) for n in cases}

    for n in cases:
        generate_cru_geometry_file(ryr[n], jsr[n], nsr[n], ttub=ttub[n], 
                                   nz=nx*3, 
                                   basename='geometries/idealized/U1_R{}.h5'.format(n))


def U1_D01234():
    nx = ny = 28 # 1.008 microns
    padding = 1
    subdivide = 3 # 36 microns to 12 microns
    
    cases = (0, 1, 2, 3, 4)
    ryr = {n: np.zeros((nx, ny)) for n in cases}
    cluster = np.ones((4, 4))

    cx = nx/2
    cy = ny/2

    for d in cases:
        lx = d/2 + d % 2
        rx = d/2


        ryr[d][cx-4-lx:cx-lx, cy-2:cy+2] += cluster
        ryr[d][cx+rx:cx+rx+4, cy-2:cy+2] += cluster

    jsr = {n: gu.create_jsr(ryr[n], padding) for n in cases}

    for d in cases:
        ryr[d] = gu.refine_ryr(ryr[d], subdivide)
        jsr[d] = gu.refine_jsr(jsr[d], subdivide, remove_edges=True)

    nsr = {d: gu.create_nsr_from_ryr(ryr[d]) for d in cases}
    ttub = {d: gu.create_ttub_from_ryr(ryr[d]) for d in cases}

    for d in cases:
        generate_cru_geometry_file(ryr[d], jsr[d], nsr[d], ttub=ttub[d], 
                                   nz=nx*3, 
                                   basename='geometries/idealized/U1_D{}_R2x16.h5'.format(d))


def U1C_D01234():
    nx = ny = 28 # 1.008 microns
    padding = 1
    subdivide = 3 # 36 microns to 12 microns
    
    cases = (0, 1, 2, 3, 4, 5)
    ryr = {n: np.zeros((nx, ny)) for n in cases}
    cluster = np.ones((4, 4))

    cx = nx/2
    cy = ny/2

    for d in cases:
        lx = d/2 + d % 2
        rx = d/2

        ryr[d][cx-4-lx:cx-lx, cy-2:cy+2] += cluster
        ryr[d][cx+rx:cx+rx+4, cy-2:cy+2] += cluster

    jsr = {n: gu.create_jsr(ryr[n], padding) for n in cases}

    for d in cases:
        ryr[d] = gu.refine_ryr(ryr[d], subdivide)
        jsr[d] = gu.refine_jsr(jsr[d], subdivide, remove_edges=True)

    nsr = {d: gu.create_nsr_from_ryr(ryr[d]) for d in cases}
    ttub = {d: gu.create_ttub_from_ryr(ryr[d]) for d in cases}

    for d in cases:
        jsr[d] = gu.contigious_jsr(jsr[d], ryr[d])


    for d in cases:
        generate_cru_geometry_file(ryr[d], jsr[d], nsr[d], ttub=ttub[d], 
                                   nz=nx*3, 
                                   basename='geometries/idealized/U1C_D{}_R2x16.h5'.format(d))


def U0123_I25():
    nx = ny = 28 # 1.008 microns
    cx = nx/2; cy = ny/2
    subdivide = 3 # 36 microns to 12 microns
    
    # Create the 5x5 cluster in the middle of the geometry
    ryr = np.zeros((nx, ny))
    ryr[cx-2:cx+3, cy-2:cy+3] = 1

    padding = (0, 1, 2, 3)
    jsr = [gu.create_jsr(ryr, p) for p in padding]
    
    ryr = gu.refine_ryr(ryr, subdivide)
    jsr = [gu.refine_jsr(j, subdivide, remove_edges=True) for j in jsr]

    nsr = gu.create_nsr_from_ryr(ryr)
    ttub = gu.create_ttub_from_ryr(ryr)

    for i, p in enumerate(padding):
        generate_cru_geometry_file(ryr, jsr[i], nsr, ttub=ttub, 
                                   nz=nx*3, 
                                   basename='geometries/idealized/U{}_R25.h5'.format(p))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #U1_G12359()
    #U1_G59121625()
    #U1_D01234()
    #U1C_D01234()
    U0123_I25()