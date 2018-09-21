import numpy as np
import vtk
import h5py
import sys

# Function to viz the RyRs location
def viz(data):

    data_matrix = np.array(data)
    data_matrix[data_matrix!=0] = 255

    nx, ny, nz = data.shape

    #for i in range(len(i0)):
    #    idx = i0[i], i1[i], i2[i]
    #    data_matrix[idx] = 255

    dataImporter = vtk.vtkImageImport()
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, nz-1, 0, ny-1, 0, nx-1)
    dataImporter.SetWholeExtent(0, nz-1, 0, ny-1, 0, nx-1)
 
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0)
    alphaChannelFunc.AddPoint(127, 0)
    alphaChannelFunc.AddPoint(128, 1)
    alphaChannelFunc.AddPoint(255, 1)

    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(0, 1.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(100, 0.0, 1.0, 0.0)
    colorFunc.AddRGBPoint(150, 0.0, 0.0, 1.0)
 
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    
    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the data for it,
    # as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    outline = vtk.vtkOutlineFilter()
    outline.SetInputConnection(dataImporter.GetOutputPort()) 
    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outline.GetOutputPort())
    outlineActor = vtk.vtkActor()
    outlineActor.SetMapper(mapOutline)
    outlineActor.GetProperty().SetColor(0, 0, 0)
 
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    
    renderer = vtk.vtkRenderer()
    renderWin = vtk.vtkRenderWindow()
    renderWin.AddRenderer(renderer)
    renderInteractor = vtk.vtkRenderWindowInteractor()
    renderInteractor.SetRenderWindow(renderWin)
    
    renderer.AddVolume(volume)
    renderer.AddViewProp(outlineActor)
    renderer.SetBackground(1,1,1)
    renderWin.SetSize(600, 600)
 
    def exitCheck(obj, event):
	if obj.GetEventPending() != 0:
	    obj.SetAbortRender(1)
 
    renderWin.AddObserver("AbortCheckEvent", exitCheck)
    renderInteractor.Initialize()
    renderWin.Render()
    renderInteractor.Start()

if __name__ == "__main__":

    if len(sys.argv) <= 1:
        print "usage python vizualise_cru_distribution.py geometry.h5"
        exit(1)

    filename = sys.argv[1]
    
    f = h5py.File(filename)
    data = f["domains"]["voxels"]

    #print data.shape
    viz(data)
