import os
import shutil
import h5py

class H5File(object):
    def __init__(self, filename):
        self.old_file = None
        self.filename = filename
        self.path, self.filename = os.path.split(filename)
        self.fullpath = filename
        
    def __enter__(self):
        # If filename already exist
        if os.path.isfile(self.fullpath):
            self.old_file = os.path.join(self.path, "_old_"+self.filename)
            shutil.move(self.fullpath, self.old_file)
        
        # Open a h5 file
        self.f = h5py.File(self.fullpath)
        return self.f 

    def __exit__(self, type, value, traceback):
        self.f.close()
        if type is None:
            if self.old_file:
                os.unlink(self.old_file)
        else:
            shutil.move(self.old_file, self.filename)


class GeometryFile(H5File):
    pass

class ParameterFile(H5File):
    pass