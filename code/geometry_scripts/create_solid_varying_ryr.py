import logging
import numpy as np
import geometry_util as gu
import matplotlib.pyplot as plt
from h5files import GeometryFile
from generate_cru_geometry_file import generate_cru_geometry_file

def generate_arrays(infile="img/1_solid", ryr_frac=1.0, width=56, jsr_layers=3, subdivide=3, ttub_mode='convexhull', ttub_pad=14, mode='nonzero'):
    indata = gu.read_ppm("img/1_solid")
    bbox = gu.trim_to_bbox(indata)
    ryr = gu.pad_array(bbox, (width, width))
    jsr = gu.create_jsr(ryr, jsr_layers)

    # Refine RyR and jSR
    ryr = gu.refine_ryr(ryr, subdivide)
    jsr = gu.refine_jsr(jsr, subdivide, remove_edges=True)
    nsr = gu.create_nsr_from_ryr(ryr)
    ttub = gu.create_ttub_from_ryr(ryr, mode=ttub_mode, pad=ttub_pad)

    original_nr_ryr = np.sum(ryr)
    nr_ryr = int(np.round(original_nr_ryr*ryr_frac))

    logging.info("RyR Fraction: {:3.2f}".format(ryr_frac))
    logging.info("Orig nr of RyR: {:3d}".format(original_nr_ryr))
    logging.info("Final nr of RyR: {:3d}".format(nr_ryr))

    gu.prune_ryrs(ryr, range(original_nr_ryr - nr_ryr), mode=mode)
    return ryr, jsr, nsr, ttub


if __name__ == '__main__':
    # np.random.seed(102030)
    # logging.getLogger().setLevel(0)

    # for m, mode in enumerate(['center']):
    #     ryrsum = np.zeros((84, 84))
    #     for ryr_frac in np.arange(1, 0, -0.1):
    #         ryr, jsr, nsr, ttub = generate_arrays(width=28, ryr_frac=ryr_frac, mode=mode) 
    #         logging.info("")

    #         # generate_cru_geometry_file(ryr, jsr, nsr, tt=ttub, nz=ryr.shape[0], cleft_size=1,
    #         #                            jsr_thickness=3, double=True, basename="geometries/var_ryr/U3_{}_RyR{}".format(mode, np.sum(ryr)))
        
    #         ryr = np.ma.masked_where(ryr == 0, ryr)
    #         plt.imshow(ryr[25:-25, 25:-25], cmap='Greys_r', interpolation='None')
    #         plt.axis('off')
    #         plt.savefig('/home/jonas/sparksim/terje/fidelity/statpdf/1solid/CR{}.png'.format(np.sum(ryr)), transparent=True)
    #         plt.close()
        
    #plt.savefig('pruning.png', bbox_inches='tight', dpi=600)


