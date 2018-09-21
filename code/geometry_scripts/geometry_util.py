import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import spatial


def read_ppm(filename):
    """Read a .ppm file and return a boolean array.

    Parameters
    ----------
    filename : str
        Name of the infile. With or without the extension.

    Returns
    -------
    numpy.ndarray
        A 2D boolean array, with shape given by input image.
    """
    assert isinstance(filename, basestring), "filename must be a string"

    # Add extension if missing
    filename = filename if filename[-4:] == ".ppm" else filename + ".ppm"

    # Read data and convert to boolean array
    data = plt.imread(filename)
    data = np.asarray(data[:, :] == 0, dtype=bool)[:, :, 0]
    return data


def trim_to_bbox(data, flip=True):
    """Trim a boolean 2D array down to its bounding box.

    Parameters
    ----------
    data : numpy.ndarray
        A boolean 2D array
    flip : bool, optional
        If true, image is transposed so first dimension is the smallest

    Returns
    -------
    numpy.ndarray
        An array of only the bounding box of the input.
    """
    assert len(data.shape) == 2, "expected 2D array"
    x, y = np.nonzero(data)
    
    # Make sure bounding box exists
    if len(x) == 0:
        raise ValueError("Input array is empty.")

    # Find bounding box
    x0, y0, x1, y1 = (min(x), min(y), max(x), max(y))
    bbox = data[x0:(x1 + 1), y0:(y1 + 1)]

    # Transpose if needed
    if flip and bbox.shape[0] > bbox.shape[1]:
        bbox = bbox.transpose()

    return bbox


def pad_array(data, final_shape):
    """Pad a 2D array with zeros to create an array of desired shape.

    First an array of the desired shape is created and initialized
    to zero, then the input array is centered in the new array.

    Parameters
    ----------
    data : numpy.ndarray
        Input array 
    shape : tuple of int
        The shape of the output

    Returns
    -------
    numpy.ndarray
        An array of the specificed output shape, 
        with the input array at the center.
    """
    x, y = final_shape
    assert len(data.shape) == 2, "expected 2D array"
    assert x >= data.shape[0], "input too large %d > %d" % (data.shape[0], x)
    assert y >= data.shape[1], "input too large %d > %d" % (data.shape[1], y)

    augmented = np.zeros((x, y), dtype=np.bool)
    dx, dy = data.shape
    x0 = (x - dx) / 2
    y0 = (y - dy) / 2
    x1 = x0 + dx
    y1 = y0 + dy
    augmented[x0:x1, y0:y1] = data
    return augmented


def create_jsr(ryr, layers):
    """Create a jSR neighborhood from RyR locations.

    jSR is assumed to be found surrounding RyR, a neighborhood
    is therefore made by first copying the RyR array and then 
    padding it with the given number of layers.

    Parameters
    ----------
    ryr : numpy.ndarray
        A 2D boolean array of RyR locations
    layers : int
        The number of jSR to pad around each RyR

    Returns
    -------
    numpy.ndarray
        An array of jSR locations, of the same shape as the input RyR array
    """
    assert len(ryr.shape) == 2, "expected 2D array"
    
    # Check there is space in array
    Nx, Ny = ryr.shape
    x, y = np.nonzero(ryr)
    if layers > min(x) or layers > min(y) or \
               layers >= Nx - min(x) or layers >= Ny - min(y):
        raise ValueError("Padding is so large the jSR will go out of bounds")

    # Pad iteratively
    jsr = ryr.copy()
    for _ in range(layers):
        jsr = morphology.binary_dilation(jsr) 
    return jsr

def contigious_jsr(jsr, ryr, mode='ryr'):
    """Ensure a jSR is contigious.

    Take in refined RyR and jSR arrays, add the convex hull of either the 
    RyRs or the jSR to make the jSR contigious.
    """
    padded_jsr = np.pad(jsr, ((10, 10), (10, 10)), mode='constant')
    cjsr = morphology.closing(padded_jsr, selem=morphology.selem.disk(10))
    jsr = (jsr + cjsr[10:-10, 10:-10]).astype(np.bool)
    return jsr



def shrink_sr(sr, layers):
    """Shrink SR by the given number of layers.

    Parameters
    ----------
    sr : numpy.ndarray
        the boolean 2D array to be shrunk
    layers : int
        number of layers to shrink

    Returns
    -------
    numpy.ndarray
        An array of the same shape as the input, with the SR shrunk.
    """
    assert len(sr.shape) == 2, "expected 2D array"
    shrunk = sr.copy()
    for i in range(1, layers+1):
        shrunk[i:, :] *= sr[:-i, :]
        shrunk[:-i, :] *= sr[i:, :]
        shrunk[:, i:] *= sr[:, :-i]
        shrunk[:, :-i] *= sr[:, i:]
    return shrunk
 
def prune_ryrs(ryr, prune_inds, mode='nonzero'):
    """Remove RyRs specified by index.

    The index of a ryr is given by the ordering
    of ``numpy.ndarray.nonzero``.

    Parameters
    ----------
    ryr : numpy.ndarray
        Boolean 2D array of RyR locations
    prune_inds : int or tuple of int
        Inidices of RyR to be pruned

    Returns
    -------
    numpy.ndarray
        An array of same shape as input, with certain RyR removed.
    """
    if mode in ['nonzero', 'scanning']:
        for ind, (x_ind, y_ind) in enumerate(zip(*ryr.nonzero())):
            if type(prune_inds) == int:
                prune_inds = range(prune_inds)
            if ind in prune_inds:
                ryr[x_ind, y_ind] = False

    elif mode in ['random', 'rand']:
        if type(prune_inds) in (tuple, list):
            prune_inds = len(prune_inds)
        ryrs = zip(*ryr.nonzero())
        np.random.shuffle(ryrs)
        for ind in range(prune_inds):
            ryr[ryrs[ind]] = False

    elif mode == 'center':
        if type(prune_inds) in (tuple, list):
            prune_inds = len(prune_inds)
        x, y = ryr.nonzero()
        cx = np.mean(x)
        cy = np.mean(y)
        ryrs = zip(x, y)

        kdtree = spatial.cKDTree([(cx, cy)])
        dist, indexes = kdtree.query(ryrs)
        dist = list(dist)

        for _ in range(prune_inds):
            ind = np.argmax(dist)
            ryr[ryrs[ind]] = False
            del dist[ind]
            del ryrs[ind]
    else:
        raise ValueError("Mode {} not understood".format(mode))

def refine_ryr(ryr, subdivide=3):
    """Increase resolution of RyR array.

    A RyR array is refined by marking only the center subpixel as a RyR. 
    
    The standard case is that the input image is given with a resolution
    of 36x36 microns per pixel. Each pixel is then split into 9 pixels of 
    size 12x12 microns.

    Parameters
    ----------
    ryr : numpy.ndarray
        A 2D boolean array denoting RyR locations, 
        most likely in 36x36 micron resolution
    subdivide : int, optional
        The number of subpixels each pixel is divided into, must be odd numbered

    Returns
    -------
    numpy.ndarray
        A 2D boolean array of larger shape than input.
    """
    assert len(ryr.shape) == 2, "expected 2D array"
    assert subdivide % 2 == 1, "have to subdivide pixels into odd nr of pixels"
    Nx, Ny = ryr.shape 
    highres = np.zeros((Nx*subdivide, Ny*subdivide), dtype=np.bool)
    for (x_ind, y_ind) in zip(*ryr.nonzero()):
        highres[x_ind*subdivide+subdivide/2, y_ind*subdivide+subdivide/2] = 1
    return highres

def refine_jsr(jsr, subdivide=3, smoothen_corners=True,
               remove_edges=False):
    """Increase resolution of jSR data.

    A jSR array is refined by marking all subpixels as jSR. For subpixels 
    belonging to a pixel not originally marked as jSR, but being located
    at a 'corner', this corner can be partially filled in to get a smoother 
    jSR after refinement. If the jSR extends to the borders of the image, 
    the outermost edge of subpixels can also be trimmed. 

    The standard case is that the input image is given with a resolution
    of 36x36 microns per pixel. Each pixel is then split into 9 pixels of 
    size 12x12 microns.

    Parameters
    ----------
    jsr : numpy.ndarray
        Array designating location of jSR
    subdivide : int, optional
        The number of subpixels each pixel is divided into, must be odd numbered
    smoothen_corners : bool, optional
        If true, corners in the refinement are smoothed 
    remove_edges : bool, optional
        Remove all jSR subpixels at the outermost edge of the array

    Returns
    -------
    numpy.ndarray2
        Athe 2D boolean array of larger shape than input.
    """
    assert len(jsr.shape) == 2, "expected 2D array"
    Nx, Ny = jsr.shape
    highres = np.zeros((Nx*subdivide, Ny*subdivide), dtype=np.bool)
    for i in range(subdivide):
        for j in range(subdivide):
            highres[i::subdivide, j::subdivide] = jsr

    if smoothen_corners:
        for _ in range(subdivide-1):
            down_diag = highres[1:, :-1]*highres[:-1, 1:]
            up_diag = highres[:-1, :-1]*highres[1:, 1:]
            
            highres[1:, 1:] += down_diag
            highres[1:, :-1] += up_diag  
            highres[:-1, 1:] += up_diag
            highres[:-1, :-1] += down_diag  

    if remove_edges:
        highres[0,:]  = 0
        highres[-1,:] = 0
        highres[:,0]  = 0
        highres[:,-1] = 0
    
    return highres

def create_nsr_columns(ryr):
    """Create the nSR column 2D template by centering them around RyR.

    The default nSR column width supposes that the RyR array has already
    been refined. 

    Parameters
    ----------
    ryr : numpy.ndarray
        A refined RyR array (12x12 microns)
    """
    assert len(ryr.shape) == 2, "expected 2D array"

    nsr = np.zeros(ryr.shape, dtype=np.bool)
    nsr[:-1, :-1] += ryr[:-1, :-1]
    nsr[:-1, :-1] += ryr[1:, :-1]
    nsr[:-1, :-1] += ryr[:-1, 1:]
    nsr[:-1, :-1] += ryr[1:, 1:]
    return nsr

def create_nsr_from_jsr(jsr, shrinkage=1):
    """Create nSR columns by thinning out jSR.

    Notes
    -----
    Requires the refined jSR.
    """
    nsr = shrink_sr(jsr, shrinkage)
    nsr[::3, :] = 0
    nsr[:, ::3] = 0
    return nsr
    
def pprint_array(data, t='1', f='0'):
    """Pretty print a boolean 2D array.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D boolean array to be printed 
    t : str, optional
        Symbol to use for true values 
    f : str, optional
        Symbol to use for false values
    """
    for row in data:
        for element in row:
            print t if element else f,
        print ""


def plot_array(data, title='', psize=36e-3, cmap='Greys_r', **kwargs):
    """Plot a 2D array.

    Parameters
    ----------
    data : numpy.ndarray
        A array to be plotted
    title : str, optional
        Title to add to the plot
    psize : float, optional
        Size of each pixel in microns, relevant for labeling of the axis
    **kwargs
        Keyword argments to be passed along to matplotlib.pyplot.imshow
    """
    m, n = data.shape
    plt.imshow(data, interpolation="none", origin="lower", cmap=cmap, 
               extent=[0, m*psize, 0, n*psize], **kwargs)
    plt.title(title)
    plt.xlabel(r"$x\ [\mathrm{\mu{}m}]$")
    plt.ylabel(r"$y\ [\mathrm{\mu{}m}]$")


def save_combined_2D_pdf(filename, ryr, jsr, nsr, psize=12e-3, cmap='Greys_r'):
    """Create a pdf of the combined 2D arrays overlaid each other.

    All arrays should be refined, i.e., 12x12 nm resolution.

    Parameters
    ----------
    filename : str
        Basename of the resulting pdf file
    ryr : numpy.ndarray
        Location of RyR
    jsr : numpy.ndarray
        Location of jSR
    nsr : numpy.ndarray
        Location of nSR
    psize : float, optional
        Size of each pixel in microns, relevant for labeling of the axis
    cmap : str, optional
        Name of matplotlib colormap to use.
    """
    combined = jsr.astype(np.uint8)
    combined += nsr
    combined += ryr

    cmap = plt.get_cmap(cmap, 4)
    plot_array(combined, psize=psize, cmap=cmap,
               title='2D arrays overlaid', vmin=0, vmax=4)

    cbar = plt.colorbar(ticks=np.arange(0.5, 5))
    cbar.ax.set_yticklabels(['Cytosol', 'jSR', 'nSR', 'RyR'])
    plt.savefig("pdf/{}_2D.pdf".format(filename))
    plt.close()

def create_ttub_from_ryr(ryr, mode='convexhull', pad=10):
    """Define a t-tubule surface based on the jSR surface.

    Notes
    -----
    The different modes decide how the t-tubule will look. Bbox makes the
    t-tubule a rectangle corresponding to the bounding box of the RyRs, which 
    is then padded in both directions. Span is like bbox in one dimension, but 
    spans the entire other dimension (use spanx/spany).
    Convex hull means the t-tubule is the convex hull of the RyRs, which is 
    then padded by repeated morphological dilation.

    Parameters
    ----------
    ryr : numpy.ndarray
        Location of ryr
    mode : str
        How t-tubule is defined: mirror/bbox/convexhull/span
    """
    assert len(ryr.shape) == 2, "expected 2D array"
    x, y = np.nonzero(ryr)
    if len(x) == 0:
        raise ValueError("Input array is empty")    

    # Find bounding box
    x0, y0, x1, y1 = (min(x),  min(y), max(x), max(y))
    
    # Create ttub
    ttub = np.zeros(ryr.shape)
    if mode == 'bbox':
        ttub[x0-pad:(x1+1+pad), y0-pad:(y1+1+pad)] = 1
    elif mode in ['convexhull', 'convex', 'hull']:
        ttub = morphology.convex_hull_image(ryr)
        for _ in range(pad):
            ttub = morphology.binary_dilation(ttub)
    elif mode in ['span', 'spany']:
        ttub[x0-pad:(x1+1+pad)] = 1
    elif mode == 'spanx':
        ttub[:, y0-pad:(y1+1+pad)] = 1
    else:
        raise ValueError("mode not understood: legal modes (bbox, span, spanx, spany, convex)")
    return ttub


def create_ttub_from_jsr(jsr, mode='convexhull'):
    """Define a t-tubule surface based on the jSR surface.

    Notes
    -----
    The different modes decide how the t-tubule will look. Mirror makes 
    the t-tubule an exact copy of the jSR, including holes. Bbox makes the
    t-tubule a rectangle corresponding to the bounding box of the jSR. 
    Convex hull means the t-tubule is the convex hull of the jSR. Span 
    means that the t-tubule is as wide as the jSR along the x-axis, but spans
    the y-axis (to mimic a cylinder).

    Parameters
    ----------
    jsr : numpy.ndarray
        Location of jSR
    mode : str
        How t-tubule is defined: mirror/bbox/convexhull/span
    """
    assert len(jsr.shape) == 2, "expected 2D array"
    x, y = np.nonzero(jsr)
    if len(x) == 0:
        raise ValueError("Input array is empty")

    # Find bounding box
    x0, y0, x1, y1 = (min(x),  min(y), max(x), max(y))

    # Create ttub
    ttub = np.zeros(jsr.shape)
    if mode == 'bbox':
        ttub[x0:(x1+1), y0:(y1+1)] = 1
    elif mode in ['convexhull', 'convex', 'hull']:
        ttub = morphology.convex_hull_image(jsr)
    elif mode == 'span':
        ttub[x0:(x1+1), :] = 1
    elif mode == 'mirror':
        ttub[jsr] = 1
    else:
        raise ValueError("mode not understood: legal modes (bbox, mirror, span, convex)")
    return ttub

def create_nsr_roof(jsr):
    """Create the nSR roof based on a jSR array.

    The nSR roof is created by taking the bounding box of each connected
    component of the jSR.
    """
    labels, n = morphology.label(jsr, background=0, connectivity=2,
                                             return_num=True)
    nsr = np.zeros(jsr.shape)
    for i in range(n):
        cluster = (labels == i+1)
        x, y = np.nonzero(cluster)
        x0, y0, x1, y1 = (min(x), min(y), max(x), max(y))
        nsr[x0:x1+1, y0:y1+1] = 1
    return nsr

if __name__ == '__main__':
    infile = 'img/GX'
    width = 28
    subdivide = 3
    
    indata = read_ppm(infile)
    bbox = trim_to_bbox(indata)
    ryr = pad_array(bbox, (width, width))

    jsr_course = create_jsr(ryr, 1)

    # # Refine RyR and jSR
    ryr = refine_ryr(ryr, subdivide)
    jsr = refine_jsr(jsr_course, subdivide, remove_edges=True)

    chjsr = contigious_jsr(jsr, ryr)

    # Find nSR column locations
    nsr = create_nsr_columns(ryr)

    # #plt.imshow(ryr, cmap='Greys_r', interpolation='none')
    plt.imshow((2*ryr + chjsr + jsr).T[::-1, ::-1], cmap='Greys_r', interpolation='none')
    plt.axis('off')
    plt.show()
