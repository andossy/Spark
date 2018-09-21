import h5py
import sys
import numpy as np
import vtk

def make_volume_files(id):
    file = open('type%d.vtk' % id ,'w')
    file.write('# vtk DataFile Version 3.0\nvtk output\nASCII\nDATASET POLYDATA\n')

    idx = np.nonzero(voxels == id)
    N = idx[0].shape[0]
            
    file.write('POINTS %d float\n' % N)
    for i in range(N):
        file.write('%d %d %d\n' % (idx[0][i], idx[1][i], idx[2][i]))

def write_boundaries(m, name):

    file = open(name+'.bnd', 'w')
    for i in range(m.shape[0]):
        p1 = m[i,:3];
        p2 = m[i,3:];
        p = (p1 + p2)/2.;
        d = abs(p2-p1).argmax(); # which dimension is changing, x, y, z?
        file.write('%f %f %f %d\n' %  (p[0],p[1],p[2],d))

    file.close()

def get_volume_actor(filename):

    reader = vtk.vtkPolyDataReader();
    reader.SetFileName(filename)
    reader.Update()


    vertexFilter = vtk.vtkGlyph3D()
    box = vtk.vtkCubeSource()
    vertexFilter.SetSourceConnection(box.GetOutputPort())
    vertexFilter.SetInputConnection(reader.GetOutputPort())
    vertexFilter.Update()


    # Create a mapper and actor for smoothed dataset
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertexFilter.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    return actor


def add_boundary(renderer, filename, color = (0.5,0.5,0.5), opacity = 1):

    m = np.loadtxt(filename)
    if len(m.shape)==1:
        m = m.reshape(1,4)
        
    centers = m[:,:3];
    d = m[:,-1];
    N = d.shape[0]
        
    for i in range(N):
        box = vtk.vtkCubeSource()
        box.SetCenter(centers[i,:])
        dim = int(d[i])
        f = 0.1
        if dim==0:
            box.SetXLength(f)
        if dim==1:
            box.SetYLength(f)
        if dim==2:
            box.SetZLength(f)
          
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(box.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        renderer.AddActor(actor)


filename = sys.argv[1]

data = h5py.File(filename);
#print data.keys()
#print data['boundaries']
#voxels = domains['voxels']

voxels  = data['domains']['voxels'].value



ids = np.unique(voxels)[1:] # discard the first type, the cytosol, from viz.

for id in ids:
    make_volume_files(id)

for key in data['boundaries'].keys():
    write_boundaries(data['boundaries'][key].value, key)




# Visualize
renderer = vtk.vtkRenderer()
renderWindow = vtk.vtkRenderWindow()
renderWindow.AddRenderer(renderer)
renderWindowInteractor = vtk.vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)
 
# Add actors and render
actor1 = get_volume_actor("type1.vtk")
actor1.GetProperty().SetColor(1,0,0)
actor2 = get_volume_actor("type2.vtk")
actor2.GetProperty().SetColor(0,1,0)
actor2.GetProperty().SetOpacity(0.5)
actor3 = get_volume_actor("type3.vtk")
actor3.GetProperty().SetColor(0,0.5,0)
actor3.GetProperty().SetOpacity(0.5)
actor4 = get_volume_actor("type4.vtk")
actor4.GetProperty().SetColor(0,0,1)
actor4.GetProperty().SetOpacity(0.03)

#add_boundary(renderer, 'serca.bnd', (1,1,0))
add_boundary(renderer, 'sr_cleft.bnd', (0,1,1))
add_boundary(renderer, 'ryr.bnd', (1,0,1))

#renderer.AddActor(actor1) # cleft
renderer.AddActor(actor2)
renderer.AddActor(actor3)
renderer.AddActor(actor4)

bb = vtk.vtkOutlineSource();
Nx, Ny, Nz = voxels.shape
bb.SetBounds(0, Nx, 0, Ny, 0, Nz);


mapper = vtk.vtkPolyDataMapper();
mapper.SetInputConnection(bb.GetOutputPort());

actor = vtk.vtkActor()
actor.SetMapper(mapper);
actor.GetProperty().SetColor((0,0,0))
renderer.AddActor(actor)

renderer.SetBackground(1, 1, 1)  # Background color white
renderWindow.SetSize(800, 800)
renderWindow.Render()


renderWindowInteractor.Start()


