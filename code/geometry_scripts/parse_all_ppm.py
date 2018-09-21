import logging
from generate_cru_geometry_file import ppm_to_geomfile

if __name__ == '__main__':
    logging.getLogger().setLevel(0)

    names = ['1_solid', '6_sub','7_sub','10_sub']
    for pad in range(5):
        for name in names:
            # 2 micron cubes
            # basename = "geometries/P%d_" % pad + name
            # ppm_to_geomfile("img/%s" % name, basename, pad, N=56, save2d_pdf=True)

            # 1 micron cubes
            basename = "geometries/U%d_" % pad + name
            try:
                ppm_to_geomfile("img/%s" % name, basename, pad, N=28, save2d_pdf=True)
            except:
                pass