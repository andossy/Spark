import numpy as np
import math
import time
import pylab
from instant import inline_with_numpy
from random import random
import os

laplace_code = """
void laplace3D (int nx0, int ny0, int nz0, double* C0,
                int nx1, int ny1, int nz1, double* C1,
                double alpha, int num_threads)
{

  // Set num threads
  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double C0_tmp;
  
  // Ghost X end sheets
  for (j=1; j<ny0-1; j++)
  {
    for (i=1; i<nx0-1; i++)
    {

      k=0;
      C0[i*nz0*ny0+j*nz0+k] = C0[i*nz1*ny1+j*nz1+k+1];

      k=nz0-1;
      C0[i*nz0*ny0+j*nz0+k] = C0[i*nz1*ny1+j*nz1+k-1];

    }
  }  
  
  // Ghost Y end sheets 
  for (i=1; i<nx0-1; i++)
  {
    for (k=1; k<nz0-1; k++)
    {
      j=0;
      C0[i*nz0*ny0+j*nz0+k] = C0[i*nz1*ny1+(j+1)*nz1+k];
  
      j=ny0-1;
      C0[i*nz0*ny0+j*nz0+k] = C0[i*nz1*ny1+(j-1)*nz1+k];
    }
  }  
  
  // Ghost Z end sheets
  for (j=1; j<ny0-1; j++)
  {
    for (k=1; k<nz0-1; k++)
    {
      i=0;
      C0[i*nz0*ny0+j*nz0+k] = C0[(i+1)*nz1*ny1+j*nz1+k];
  
      i=nx0-1;
      C0[i*nz0*ny0+j*nz0+k] = C0[(i-1)*nz1*ny1+j*nz1+k];
    }
  }  
  
  // Main kernel loop
  #pragma omp parallel for private(i, j, k, C0_tmp) collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        C0_tmp = -6*C0[i*nz0*ny0+j*nz0+k] + \
           C0[(i-1)*nz0*ny0+j*nz0+k] + C0[(i+1)*nz0*ny0+j*nz0+k] + \
           C0[i*nz0*ny0+(j-1)*nz0+k] + C0[i*nz0*ny0+(j+1)*nz0+k] + \
           C0[i*nz0*ny0+j*nz0+k-1] + C0[i*nz0*ny0+j*nz0+k+1];
  
        // Put value back into return array with offset to indices 
        C1[i*nz1*ny1+j*nz1+k] = C0[i*nz1*ny1+j*nz1+k] + C0_tmp*alpha;
      }
    }  
  }    
}
"""

reaction_code = """
void reaction3D (int nx0, int ny0, int nz0, double* Ca,
                 int nx1, int ny1, int nz1, double* buff,
                 double B_tot, double k_on, double k_off, double dt, int num_threads)
{

  // Set num threads
  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double J;

  // Use pointers reducing indexing into memory to once 
  double* Ca_ijk;
  double* buff_ijk;
  
  // Main kernel loop
  #pragma omp parallel for private(i, j, k, J, Ca_ijk, buff_ijk) collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        Ca_ijk = &Ca[i*nz0*ny0+j*nz0+k];
        buff_ijk = &buff[i*nz0*ny0+j*nz0+k];
        J = k_on*(B_tot - *buff_ijk)*(*Ca_ijk) - \
            k_off*(*buff_ijk);
        *Ca_ijk -= dt*J;
        *buff_ijk += dt*J;
        
      }
    }  
  }    
}
"""

serca_code = """
void serca3D (int nx0, int ny0, int nz0, double* Ca_i,
              int nx1, int ny1, int nz1, double* Ca_SR,
              double dt, double gamma, double fudge, int num_threads)
{

  // Set num threads
  omp_set_num_threads(num_threads);

  // Local variables
  int i, j, k;
  double J;

  // Use pointers reducing indexing into memory to once 
  double Ca_i2_ijk;
  double Ca_SR2_ijk;
  
  // Main kernel loop
  #pragma omp parallel for private(i, j, k, J, Ca_i2_ijk, Ca_SR2_ijk) collapse(3)
  for (i=1; i<nx0-1; i++)
  {
    for (j=1; j<ny0-1; j++)
    {
      for (k=1; k<nz0-1; k++)
      {
        // Main kernel
        Ca_i2_ijk = Ca_i[i*nz0*ny0+j*nz0+k];
        Ca_SR2_ijk = Ca_SR[i*nz0*ny0+j*nz0+k];
        Ca_i2_ijk *= Ca_i2_ijk;
        Ca_SR2_ijk *= Ca_SR2_ijk;
        J = fudge*(570997802.885875*Ca_i2_ijk - 0.0425239333622699*Ca_SR2_ijk)/(106720651.206402*Ca_i2_ijk + 182.498197548666*Ca_SR2_ijk + 5.35062954944879);
        Ca_i[i*nz0*ny0+j*nz0+k] -= dt*J;
        Ca_SR[i*nz0*ny0+j*nz0+k] += dt*J/gamma;
      }
    }  
  }    
}
"""


laplace3D = inline_with_numpy(laplace_code, arrays = [["nx0", "ny0", "nz0", "C0"],
                                                      ["nx1", "ny1", "nz1", "C1"]],
                              system_headers=["omp.h"], cppargs=["-O3", "-fopenmp"],
                              lddargs=["-fopenmp"])

reaction3D = inline_with_numpy(reaction_code, arrays = [["nx0", "ny0", "nz0", "Ca"],
                                                        ["nx1", "ny1", "nz1", "buff"]],
                               system_headers=["omp.h"], cppargs=["-O3", "-fopenmp"],
                               lddargs=["-fopenmp"])

serca3D = inline_with_numpy(serca_code, arrays = [["nx0", "ny0", "nz0", "Ca_i"],
                                                  ["nx1", "ny1", "nz1", "Ca_SR"]],
                            system_headers=["omp.h"], cppargs=["-O3", "-fopenmp"],
                            lddargs=["-fopenmp"])

def load_indices(nx, ny, nz, h):

    # Scale nx, xy, nz in terms of RyR
    h_scale = 30/h
    nx = int(nx/h_scale)
    ny = int(ny/h_scale)
    nz = int(nz/h_scale)

    # All CaRU placed mid-sarcomere
    mid_x = (nx+1)/2;

    # load RyR indices from file
    i1 = np.fromfile(open("i_RyR_indices.csv"), sep=", ", dtype=int)
    i2 = np.fromfile(open("j_RyR_indices.csv"), sep=", ", dtype=int)

    # Only use the subset which are inside the geometry
    #print "num RyR before reduction:", len(i1)
    i1 = i1[i1<ny]
    i2 = i2[i1<ny]
    i1_ryr = i1[i2<nz]
    i2_ryr = i2[i2<nz]

    # Scale indices and move to center of macro voxel
    i1_ryr = i1_ryr*h_scale - math.floor(h_scale/2)
    i2_ryr = i2_ryr*h_scale - math.floor(h_scale/2)
    
    i0_ryr = np.ones(len(i1_ryr), dtype=int)*mid_x*h_scale

    # load CSQN indices from file
    i1 = np.fromfile(open("i_csqn_indices.csv"), sep=", ", dtype=int)
    i2 = np.fromfile(open("j_csqn_indices.csv"), sep=", ", dtype=int)

    # Only use the subset which are inside the geometry
    i1 = i1[i1<ny]
    i2 = i2[i1<ny]
    i1_csqn = i1[i2<nz]*h_scale
    i2_csqn = i2[i2<nz]*h_scale
    i0_csqn = np.ones(len(i1_csqn), dtype=int)*mid_x*h_scale

    # Add CSQN to all voxels covered by the original CSQN array
    if h_scale > 1:

        i0_csqn_list = []
        i1_csqn_list = []
        i2_csqn_list = []
        
        # Add offsetted versions of the csqn
        for i in range(h_scale):
            for j in range(h_scale):
                i0_csqn_list.append(i0_csqn)
                i1_csqn_list.append(i1_csqn+i)
                i2_csqn_list.append(i2_csqn+j)

        i0_csqn = np.concatenate(i0_csqn_list)
        i1_csqn = np.concatenate(i1_csqn_list)
        i2_csqn = np.concatenate(i2_csqn_list)

    return i0_ryr, i1_ryr, i2_ryr, i0_csqn, i1_csqn, i2_csqn

# Function to viz the RyRs location
def viz(i0, i1, i2, nx, ny, nz):

    import vtk
    data_matrix = np.zeros([nx+2, ny+2, nz+2], dtype=np.uint8)

    for i in range(len(i0)):
	idx = i0[i], i1[i], i2[i]
	data_matrix[idx] = 255

    dataImporter = vtk.vtkImageImport()
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    dataImporter.SetDataScalarTypeToUnsignedChar()
    dataImporter.SetNumberOfScalarComponents(1)
    dataImporter.SetDataExtent(0, nz+1, 0, ny+1, 0, nx+1)
    dataImporter.SetWholeExtent(0, nz+1, 0, ny+1, 0, nx+1)
 

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
    # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
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

h = 30;
Lx = 30*50;
Ly = 30*100;
Lz = 30*70;

assert h <= 30,"h needs to be equal or smaller than the size of a RyR"
assert 30 % h == 0,"There cannot be any rest when dividing the size of a RyR with h"

nx = 1+int(math.ceil(Lx/h));
ny = 1+int(math.ceil(Ly/h));
nz = 1+int(math.ceil(Lz/h));

print nx, ny, nz
one = np.ones((nx+2,ny+2,nz+2))

Vfraction = (30./h)**3; # scaling of RyR when changing dx

# Define where the RyRs are:
# first set the numbers of RyR in a CaRU;
w = int(round(60./h));  # the number depends on the resolution
n_sq = 2*w+1;
n_ryr = n_sq**2;

# All CaRU placed mid-sarcomere
mid_x = (nx+1)/2;

# Load precomputed indices from file
i0_ryr, i1_ryr, i2_ryr, i0_csqn, i1_csqn, i2_csqn = load_indices(nx, ny, nz, h)

n_ryr = i0_ryr.shape[0]
viz(i0_ryr, i1_ryr, i2_ryr, nx, ny, nz)
viz(i0_csqn, i1_csqn, i2_csqn, nx, ny, nz)

fig_size =  [12,6]
params = {'figure.figsize': fig_size}
pylab.rcParams.update(params)

save_data = False

if save_data:
    dirname = time.strftime('%m%d%H%M')
    root = os.environ['HOME']+"/data/"+dirname+"/" # where to save data
    try:
	os.mkdir(root)
    except OSError:
	print "Directory exists."

    print "Will save to ", root

# Set constants and dt based on these:
D_i = 220e3;
D_SR = 73.3e3;
D_ATP = 140e3;
D_CMDN = 22e3;
D_Fluo = 42e3;

dt = (1./6)*h**2/D_i;

alpha_i = dt*D_i/h**2;
Ca0 = 140e-3;
Ca_i = Ca0*one;

alpha_SR = dt*D_SR/h**2;
Ca_SR = 1.3e3*one;

k_on_CMDN = 34e-3;
k_off_CMDN = 238e-3;
CMDN_tot = 24;
alpha_CMDN = dt*D_CMDN/h**2;

k_on_ATP = 255e-3;
k_off_ATP = 45;
ATP_tot = 455;
alpha_ATP = dt*D_ATP/h**2;

k_on_Fluo = 110e-3;
k_off_Fluo = 110e-3;
Fluo_tot = 25;
alpha_Fluo = dt*D_Fluo/h**2;

k_on_TRPN = 32.7e-3;
k_off_TRPN = 19.6e-3;
TRPN_tot = 70;

k_on_CSQN = 102e-3;
k_off_CSQN = 65
CSQN_tot = 30e3;

# Manually set IC:
#Ca_ATP0 = 318e-3
#Ca_CMDN0 = 471e-3
#Ca_Fluo0 = 2.82
#Ca_TRPN0 = 13.2
#print Ca_ATP0, Ca_CMDN0, Ca_Fluo0, Ca_TRPN0

alpha = [alpha_i, alpha_SR, alpha_CMDN, alpha_ATP, alpha_Fluo, 0, 0]
k_on =  [0,       0,        k_on_CMDN,  k_on_ATP,  k_on_Fluo,  k_on_TRPN, k_on_CSQN]
k_off = [0,       0,        k_off_CMDN, k_off_ATP, k_off_Fluo, k_off_TRPN, k_off_CSQN]
B_tot = [0,       0,        CMDN_tot,   ATP_tot,   Fluo_tot,   TRPN_tot, CSQN_tot]

# Calculate steady state IC for the buffers based on Ca_i ...
Ca_CMDN0, Ca_ATP0, Ca_Fluo0, Ca_TRPN0 = (np.array(B_tot)*Ca0/(Ca0+np.array(k_off)/np.array(k_on)))[2:-1]
# and Ca_SR:
Ca_CSQN0 = CSQN_tot*Ca_SR[0,0,0]/(Ca_SR[0,0,0] +  k_off_CSQN/k_on_CSQN)

print Ca_ATP0, Ca_CMDN0, Ca_Fluo0, Ca_TRPN0, Ca_CSQN0

# Allocate the data structure for the solution
Ca_ATP  = Ca_ATP0*one
Ca_CMDN = Ca_CMDN0*one
Ca_Fluo = Ca_Fluo0*one
Ca_TRPN = Ca_TRPN0*one
Ca_CSQN = Ca_CSQN0*one

Ca = [[Ca_i.copy(),    Ca_i   ],
      [Ca_SR.copy(),   Ca_SR  ],
      [Ca_CMDN.copy(), Ca_CMDN],
      [Ca_ATP.copy(),  Ca_ATP ],
      [Ca_Fluo.copy(), Ca_Fluo],
      [Ca_TRPN, Ca_TRPN],
      [Ca_CSQN, Ca_CSQN]]

gamma = 0.02; # SR volume fraction
cai, sri, cmdni, atpi, fluoi, trpni, csqni = range(len(Ca))

fraction = [1, gamma, 1, 1, 1, 1, gamma];

# Ryr conductance:
k = (Vfraction)*150/2; # 1/ms, based on 0.5pA of Ca2+ into (30nm)^3.
K = np.exp(-k*dt*(1+1/gamma)); # factor need in the integration below

# Initial states of the RyRs
states = np.zeros((2, n_ryr));
#states[0,12] = 1;
states[0,1:23:3] = 1;


#I = np.arange(0,n,dtype=int)
#Im = np.arange(-1,n-1,dtype=int); Im[0] = 1;
#Ip = np.arange(1,n+1,dtype=int);  Ip[n-1] = n-2
#J = 0*one;

def diffusion_step(U, alpha):

    U0, U1 = U

    # Mirror values at the end to enforce no-flow (and mass conservation)
    U0[0,:,:]   =  U0[1,:,:]
    U0[nx+1,:,:] = U0[nx,:,:]
    U0[:,0,:]   =  U0[:,1,:]
    U0[:,ny+1,:] = U0[:,ny,:]
    U0[:,:,0]   =  U0[:,:,1]
    U0[:,:,nz+1] = U0[:,:,nz]

    # Exp Euler step:
    U1[1:nx+1,1:ny+1,1:nz+1] = U0[1:nx+1,1:ny+1,1:nz+1] + \
                               alpha*(-6*U0[1:nx+1,1:ny+1,1:nz+1]+\
                                      U0[0:nx,1:ny+1,1:nz+1]+\
                                      U0[2:nx+2,1:ny+1,1:nz+1]+\
                                      U0[1:nx+1,0:ny,1:nz+1]+\
                                      U0[1:nx+1,2:ny+2,1:nz+1]+\
                                      U0[1:nx+1,1:ny+1,0:nz]+\
                                      U0[1:nx+1,1:ny+1,2:nz+2])
    
def reaction_step(Ca, buff, B_tot, k_on, k_off):
    J = k_on*(B_tot - buff)*Ca - k_off*buff;
    Ca -= dt*J;
    buff += dt*J;

def serca_step(Ca_i, Ca_SR):
    fudge = 1
    J_Serca = fudge*(570997802.885875*Ca_i**2 - 0.0425239333622699*Ca_SR**2)/(106720651.206402*Ca_i**2 + 182.498197548666*Ca_SR**2 + 5.35062954944879)
    Ca_i -= dt*J_Serca
    Ca_SR += dt*J_Serca/gamma;
    
def update_ryr(Ca_i, Ca_SR, Ca_CSQN):
    # CSQN step:
    idx_csqn = i0_csqn, i1_csqn, i2_csqn
    J = k_on_CSQN*(CSQN_tot - Ca_CSQN[idx_csqn])*Ca_SR[idx_csqn] - \
        k_off_CSQN*Ca_CSQN[idx_csqn]
    Ca_SR[idx_csqn] -= dt*J;
    Ca_CSQN[idx_csqn] += dt*J;

    for i, idx in enumerate(zip(i0_ryr, i1_ryr, i2_ryr)):
	#idx = i0_ryr[i], i1_ryr[i], i2_ryr[i]
	#Continous formulation
	#states[:,i] += dt*stern(t, states[:,i], Ca_i[idx])
	states[:,i] = stern_discrete(dt, states[:,i], Ca_i[idx])
	open = states[0,i]*(1-states[1,i])
	#Exp Euler:
	#J_RyR = k*open*(Ca_SR[idx]-Ca_i[idx])
	#Ca_i[idx]  += dt*J_RyR
	#Ca_SR[idx] -= dt*J_RyR/gamma;
	#Analytical update:
	if open:
	    c0 = (Ca_i[idx] + gamma*Ca_SR[idx])/(1+gamma);
	    c1 = (Ca_i[idx] - Ca_SR[idx])/(1+1/gamma);
	    Ca_i[idx] =  c0 + c1*K
	    Ca_SR[idx] = c0 - c1*K/gamma;
            

def stern(t,y,Ca):
    m = y[0];
    h = y[1];
    kim = 0.005;
    kom = 0.06;
    K_i = 0.01*10;
    K_o = 0.01*41.4;
    ki = kim/K_i;
    ko = kom/K_o**2;
    dm = ko*Ca**2*(1-m) - kom*m;
    dh = ki*Ca*(1-h) - kim*h;
    
    return np.array((dm,dh))

def stern_discrete(dt,y,Ca):

    kim = 0*0.005;
    kom = 0*0.06;
    ki = Ca*0.5;
    ko = 1e-2*Ca**2*35

    m = y[0];
    if m==1:
	r = random()
	m = 1 - (r<dt*kom)
    else:
	m = 1*(random()<dt*ko)
	
    h = y[1];
    if h==1:
	h = 1 - (random()<dt*kim)
    else:
	h = 1*(random()<dt*ki)

    return np.array((m,h))



T = 10
t = 0
counter = 0;
mean = [];
DT = 0.01; # plotting time step

use_instant = 1
num_threads = 0
t00 = time.time()
t_ryr = 0.0
t_conc = 0.0
while t<T:
    t += dt
    t_conc_0 = time.time()

    # Diffusion steps
    for i in range(5):
        if use_instant:
            laplace3D(Ca[i][0], Ca[i][1], alpha[i], num_threads)
        else:
            diffusion_step(Ca[i], alpha[i])

    # Buffer steps
    for i in range(2,6):
        if use_instant:
            reaction3D(Ca[cai][1], Ca[i][1], B_tot[i], k_on[i], k_off[i], dt, num_threads)
        else:
            reaction_step(Ca[cai][1], Ca[i][1], B_tot[i], k_on[i], k_off[i])

    # Serca step:
    if use_instant:
        serca3D(Ca[cai][1], Ca[sri][1], dt, gamma, 1.0, num_threads)
    else:
        serca_step(Ca[cai][1], Ca[sri][1])

    # Update at RyRs, one at the time
    t_ryr_0 = time.time()
    t_conc += t_ryr_0 - t_conc_0
    update_ryr(Ca[cai][1], Ca[sri][1], Ca[csqni][1])
    t_ryr += time.time() - t_ryr_0

    if math.fmod(t,DT)<dt:
        t_io_0 = time.time()
	sm = 0;
	ca = [t]
	for i in range(7):
	    sum_c_i = Ca[i][1][1:nx+1,1:ny+1,1:nz+1].sum()
	    sm += fraction[i]*sum_c_i
	    ca += [sum_c_i/(nx*ny*nz)]

	mean += [ca];
        print
	print counter, t, ca[1], ca[2], Ca[cai][1].min(), Ca[cai][1].max(), sm
	np.savetxt("mean%02d.txt"%h,np.array(mean))
        
    pylab.interactive(True)
    if math.fmod(t,DT)<dt:
	pylab.clf()
	pylab.subplot(121)
	pylab.imshow(Ca[cai][1][mid_x,1:-1,1:-1])
	pylab.clim([0.1,20])

	pylab.subplot(122)
	pylab.imshow(Ca[cai][1][mid_x,1:-1,1:-1])
	pylab.clim([0.1,1])
	pylab.draw()
	if save_data:
	    for i in range(7):
		Ca[i][1].dump(root+"Ca%d_T%04d.np"%(i,counter))
	    counter += 1

        t11 = time.time()
        print "timing:", t11-t00, "with instant" if use_instant else "without instant"
        print "Conc timing:", t_conc, "with instant" if use_instant else "without instant"
        print "RyR timing:", t_ryr
        print "IO timing:", time.time()-t_io_0
        t00 = t11
        t_ryr = 0.0
        t_conc = 0.0

    # Update Ca
    for i in range(len(Ca)):
        Cai0, Cai1 = Ca[i]
        Ca[i] = [Cai1, Cai0]

