#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import numpy

# Typde defs
domain_id = numpy.uint8
float_type = numpy.float32
float_type = numpy.float64


class OuterBoundary(object):
    def __init__(self, type, species, geom_size, geom_h, **kwargs):
        assert isinstance(geom_size, tuple)
        self.name = self.__class__.__name__.lower()
        self.boundary = self.name
        self.constants = kwargs

        assert type in ['discrete', 'density'], 'Type must be one of [density, discrete]'
        self.type = type

        assert isinstance(species, str)
        self.species = species

        num_voxels = map(lambda size: size/geom_h, geom_size)
        self.data = numpy.zeros(num_voxels, dtype=int)

    def voxels(self):
        ''' Returns voxels as a 2D array of shape Nx6 '''
        X,Y,Z = numpy.nonzero(self.data)
        voxels = [[x,y,z,x,y,z] for x,y,z in zip(X,Y,Z)]
        return numpy.asarray(voxels, dtype=numpy.uint16)

    def check_consistency(self):
        '''
        Return True if the data array contains 0-1 values only. All values inside
        the array (i.e. indices 1...N-1) must be zero. False otherwise '''
        return (numpy.unique(self.data) == [0,1]).all() and (numpy.unique(self.data[1:-1, 1:-1, 1:-1]) == [0]).all()

    def __repr__(self):
        return 'OuterBoundary(name=%s)' % self.name


class Drain(OuterBoundary):
    def __init__(self, geom_size, geom_h):
        super(Drain, self).__init__('density', 'Ca', geom_size, geom_h, scale=2000.0, shift=0.14)

        for i in xrange(3):
            self.data[tuple(numpy.roll(numpy.index_exp[0,:,:], i))] = 1
            self.data[tuple(numpy.roll(numpy.index_exp[-1,:,:], i))] = 1


def check_boundaries(*boundaries):
    ''' Check consistency of boundaries. Verifies overlapping boundaries too. '''
    nb = len(boundaries)
    assert nb>0, 'No boundaries to check for consistency'

    data = reduce(lambda x,y: x+y.data, boundaries, numpy.zeros_like(boundaries[0]))
    return (numpy.unique(data) == [0,nb]).all() and all(b.check_consistency() for b in boundaries)


def append_boundaries(geometry_filename, parameters_filename, *boundaries):
    assert check_boundaries(*boundaries), 'Outer boundaries are not consistent'

    with h5py.File(geometry_filename, 'a') as fh:
        faces = fh["boundaries"]
        # Read number of stored boundaries
        nb = faces.attrs['num']

        for bi, boundary in enumerate(boundaries, nb):
            faces.attrs.create("name_%d" % bi, boundary.name)
            faces.attrs.create("type_%d" % bi, boundary.type)
            faces.create_dataset(boundary.name, data=boundary.voxels(),
                                 compression="gzip", dtype=numpy.uint16)

        faces.attrs.modify('num', bi+1)

    with h5py.File(parameters_filename, 'a') as fh:
        for boundary in boundaries:
            fh.attrs.create("use_%s" % boundary.name, True, dtype=domain_id)

            gh = fh.create_group(boundary.name)
            gh.attrs.create("boundary", boundary.boundary)
            gh.attrs.create("species", boundary.species)

            for key, value in boundary.constants.iteritems():
                gh.attrs.create(key, value, dtype=float_type)

    print "New outer boundaries have been appedned. See %r, %r" % (geometry_filename, parameters_filename)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='A helper script that appends additional information '\
                                        'on outer boundaries to both geometry and parameters files')
    parser.add_argument("geometry", metavar='GEOM', type=str,
                        help='a valid HDF5 file where a geoemtry is stored')
    parser.add_argument("parameters", metavar='PARAM', type=str,
                        help='a valid HDF5 file where parameters are stored')
    parser.add_argument("--overwrite", action='store_true', help='Force to overwrite the files. '
                        'By default new files will be created')

    args = parser.parse_args()

    import shutil
    prefix = ""
    if not args.overwrite:
        prefix = "outer_"
        shutil.copy(args.geometry, prefix+args.geometry)
        shutil.copy(args.parameters, prefix+args.parameters)

    with h5py.File(prefix+args.geometry, 'r') as fh:
        geom_size = tuple(fh.attrs["global_size"])
        geom_h = fh.attrs["h"]

    append_boundaries(prefix+args.geometry, prefix+args.parameters,
                      Drain(geom_size, geom_h))
