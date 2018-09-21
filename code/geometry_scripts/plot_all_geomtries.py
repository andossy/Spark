"""
Save a png image of all geometries in the geometries folder.
"""

from visualize_geometry import GeometryVisualizer

for name in ['1_solid', '4_sub', '6_sub','7_sub','10_sub']:
    for pad in range(1, 2):
        infile = "geometries/P{}_{}_double.h5".format(pad, name)
        outfile_front = "geometries/img/P{}_{}.png".format(pad, name)
        outfile_back  = "geometries/img/P{}_{}_back.png".format(pad, name)

        geomviz = GeometryVisualizer(infile)
        geomviz.style['domains'] = [2, 3]
        geomviz.style['boundaries'] = ['ryr']
        geomviz.style['labels'] = []  
        geomviz.style['bbox'] = False

        geomviz.imgpath = outfile_front
        geomviz.render(elevation=10, azimuth=200)

        # geomviz.imgpath=outfile_back
        # geomviz.style['domains'] = [2, 3, 4]
        # geomviz.render(elevation=120)

