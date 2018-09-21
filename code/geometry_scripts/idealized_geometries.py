from generate_cru_geometry_file import generate_cru_geometry_file
import numpy as np
import geometry_util as gu
import logging

logging.getLogger().setLevel(0)


# for p in range(5):
#     nx = ny = 56 # x36 nm = 2.016 microns

#     ryr1 = np.zeros((nx, ny))
#     ryr2 = np.zeros((nx, ny))
#     ryr3 = np.zeros((nx, ny))
#     ryr5 = np.zeros((nx, ny))
#     ryr9 = np.zeros((nx, ny))

#     ryr1[nx/2, ny/2] = 1
#     ryr2[nx/2-1:nx/2+1, ny/2] = 1
#     ryr3[nx/2-1:nx/2+2, ny/2] = 1
#     ryr5[nx/2-1:nx/2+2, ny/2] = 1
#     ryr5[nx/2, ny/2-1:ny/2+2] = 1
#     ryr9[nx/2-1:nx/2+2, ny/2-1:ny/2+2] = 1


#     jsr = gu.create_jsr(ryr9, p)

#     # Refine RyR and jSR
#     ryr1 = gu.refine_ryr(ryr1, 3)
#     ryr2 = gu.refine_ryr(ryr2, 3)
#     ryr3 = gu.refine_ryr(ryr3, 3)
#     ryr5 = gu.refine_ryr(ryr5, 3)
#     ryr9 = gu.refine_ryr(ryr9, 3)
#     jsr = gu.refine_jsr(jsr, 3, remove_edges=True)

#     assert(np.sum(ryr1) == 1)
#     assert(np.sum(ryr2) == 2)
#     assert(np.sum(ryr3) == 3)
#     assert(np.sum(ryr5) == 5)
#     assert(np.sum(ryr9) == 9)

#     nsr = gu.create_nsr_from_ryr(ryr9)
#     ttub = jsr.copy()

#     for r, ryr in enumerate((ryr1, ryr2, ryr3, ryr5, ryr9)):
#         generate_cru_geometry_file(ryr, jsr, nsr, tt=ttub, nz=168,
#                                    nsr_height=None, sr_volume=None,
#                                    cleft_size=1, jsr_thickness=3,
#                                    double=True, basename="geometries/idealized/P{}_T{}".format(p, [1,2,3,5,9][r]))


for p in range(5):
    nx = ny = 28 # x36 nm = 2.016 microns

    ryr1 = np.zeros((nx, ny))
    ryr2 = np.zeros((nx, ny))
    ryr3 = np.zeros((nx, ny))
    ryr5 = np.zeros((nx, ny))
    ryr9 = np.zeros((nx, ny))

    ryr1[nx/2, ny/2] = 1
    ryr2[nx/2-1:nx/2+1, ny/2] = 1
    ryr3[nx/2-1:nx/2+2, ny/2] = 1
    ryr5[nx/2-1:nx/2+2, ny/2] = 1
    ryr5[nx/2, ny/2-1:ny/2+2] = 1
    ryr9[nx/2-1:nx/2+2, ny/2-1:ny/2+2] = 1


    jsr = gu.create_jsr(ryr9, p)

    # Refine RyR and jSR
    ryr1 = gu.refine_ryr(ryr1, 3)
    ryr2 = gu.refine_ryr(ryr2, 3)
    ryr3 = gu.refine_ryr(ryr3, 3)
    ryr5 = gu.refine_ryr(ryr5, 3)
    ryr9 = gu.refine_ryr(ryr9, 3)
    jsr = gu.refine_jsr(jsr, 3, remove_edges=True)

    assert(np.sum(ryr1) == 1)
    assert(np.sum(ryr2) == 2)
    assert(np.sum(ryr3) == 3)
    assert(np.sum(ryr5) == 5)
    assert(np.sum(ryr9) == 9)

    nsr = gu.create_nsr_from_ryr(ryr9)
    ttub = jsr.copy()

    for r, ryr in enumerate((ryr1, ryr2, ryr3, ryr5, ryr9)):
        generate_cru_geometry_file(ryr, jsr, nsr, tt=ttub, nz=nx*3,
                                   nsr_height=36, sr_volume=None,
                                   cleft_size=1, jsr_thickness=3,
                                   double=True, basename="geometries/idealized/U{}_R{}".format(p, [1,2,3,5,9][r]))

