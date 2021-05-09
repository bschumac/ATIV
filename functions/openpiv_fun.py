
import numpy.lib.stride_tricks
import numpy as np
from numpy import ma
from numpy import log




def get_coordinates(image_size, window_size, overlap):
    """Compute the x, y coordinates of the centers of the interrogation windows.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size: int
        the size of the interrogation windows.
    overlap: inta
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    x : 2d np.ndarray
        a two dimensional array containing the x coordinates of the
        interrogation window centers, in pixels.
    y : 2d np.ndarray
        a two dimensional array containing the y coordinates of the
        interrogation window centers, in pixels.
    """

    # get shape of the resulting flow field
    field_shape = get_field_shape(image_size, window_size, overlap)

    # compute grid coordinates of the interrogation window centers
    x = (np.arange(field_shape[1]) * (window_size -
                                     overlap) + (window_size - 1) / 2.0) +((window_size -overlap)/2)
    y = (np.arange(field_shape[0]) * (window_size -
                                     overlap) + (window_size - 1) / 2.0)+((window_size -overlap)/2)

    return np.meshgrid(x, y[::-1])


def get_field_shape(image_size, window_size, overlap):
    """Compute the shape of the resulting flow field.
    Given the image size, the interrogation window size and
    the overlap size, it is possible to calculate the number
    of rows and columns of the resulting flow field.
    Parameters
    ----------
    image_size: two elements tuple
        a two dimensional tuple for the pixel size of the image
        first element is number of rows, second element is
        the number of columns.
    window_size: int
        the size of the interrogation window.
    overlap: int
        the number of pixel by which two adjacent interrogation
        windows overlap.
    Returns
    -------
    field_shape : two elements tuple
        the shape of the resulting flow field
    """

    return ((image_size[0] - window_size) // (window_size - overlap) + 1,
            (image_size[1] - window_size) // (window_size - overlap) + 1)



def get_org_data(frame_a, search_area_size, overlap):
    n_rows, n_cols = get_field_shape(frame_a.shape, search_area_size, overlap )    
    frame_a_org = np.zeros((n_rows, n_cols))
   
    
    for k in range(n_rows):
        # range(range(search_area_size/2, frame_a.shape[0] - search_area_size/2, window_size - overlap ):
        for m in range(n_cols):
            il = k*(search_area_size - overlap)
            ir = il + search_area_size*0.5
            
            # same for top-bottom
            jt = m*(search_area_size - overlap)
            jb = jt + search_area_size*0.5
            
            # pick up the window in the second image
            frame_a_org[k,m] = frame_a[int(ir),int(jb)]
    return frame_a_org
        






def moving_window_array(array, window_size, overlap):
    """
    This is a nice numpy trick. The concept of numpy strides should be
    clear to understand this code.
    Basically, we have a 2d array and we want to perform cross-correlation
    over the interrogation windows. An approach could be to loop over the array
    but loops are expensive in python. So we create from the array a new array
    with three dimension, of size (n_windows, window_size, window_size), in which
    each slice, (along the first axis) is an interrogation window.
    """
    sz = array.itemsize
    shape = array.shape
    array = np.ascontiguousarray(array)

    strides = (sz * shape[1] * (window_size - overlap),
               sz * (window_size - overlap), sz * shape[1], sz)
    shape = (int((shape[0] - window_size) / (window_size - overlap)) + 1, int(
        (shape[1] - window_size) / (window_size - overlap)) + 1, window_size, window_size)

    return numpy.lib.stride_tricks.as_strided(array, strides=strides, shape=shape).reshape(-1, window_size, window_size)


def find_first_peak(corr):
    """
    Find row and column indices of the first correlation peak.
    Parameters
    ----------
    corr : np.ndarray
        the correlation map
    Returns
    -------
    i : int
        the row index of the correlation peak
    j : int
        the column index of the correlation peak
    corr_max1 : int
        the value of the correlation peak
    """
    ind = corr.argmax()
    s = corr.shape[1]

    i = ind // s
    j = ind % s

    return i, j, corr.max()


def find_second_peak(corr, i=None, j=None, width=2):
    """
    Find the value of the second largest peak.
    The second largest peak is the height of the peak in
    the region outside a 3x3 submatrxi around the first
    correlation peak.
    Parameters
    ----------
    corr: np.ndarray
          the correlation map.
    i,j : ints
          row and column location of the first peak.
    width : int
        the half size of the region around the first correlation
        peak to ignore for finding the second peak.
    Returns
    -------
    i : int
        the row index of the second correlation peak.
    j : int
        the column index of the second correlation peak.
    corr_max2 : int
        the value of the second correlation peak.
    """

    if i is None or j is None:
        i, j, tmp = find_first_peak(corr)

    # create a masked view of the corr
    tmp = corr.view(ma.MaskedArray)

    # set width x width square submatrix around the first correlation peak as masked.
    # Before check if we are not too close to the boundaries, otherwise we
    # have negative indices
    iini = max(0, i - width)
    ifin = min(i + width + 1, corr.shape[0])
    jini = max(0, j - width)
    jfin = min(j + width + 1, corr.shape[1])
    tmp[iini:ifin, jini:jfin] = ma.masked
    i, j, corr_max2 = find_first_peak(tmp)

    return i, j, corr_max2


def find_subpixel_peak_position(corr, subpixel_method='gaussian'):
    """
    Find subpixel approximation of the correlation peak.
    This function returns a subpixels approximation of the correlation
    peak by using one of the several methods available. If requested,
    the function also returns the signal to noise ratio level evaluated
    from the correlation map.
    Parameters
    ----------
    corr : np.ndarray
        the correlation map.
    subpixel_method : string
         one of the following methods to estimate subpixel location of the peak:
         'centroid' [replaces default if correlation map is negative],
         'gaussian' [default if correlation map is positive],
         'parabolic'.
    Returns
    -------
    subp_peak_position : two elements tuple
        the fractional row and column indices for the sub-pixel
        approximation of the correlation peak.
    """
    # initialization
    # default_peak_position = (np.floor(corr.shape[0] / 2.), np.floor(corr.shape[1] / 2.))
    default_peak_position = (0,0)
     
    # the peak locations i = y j = x
    peak1_i, peak1_j, dummy = find_first_peak(corr)
    
    # if the peak is in the corner of corr then it needs to be moved inside 1 px
    
    if peak1_i == 0:
        peak1_i += 1
    if peak1_j == 0:
        peak1_j += 1
    if peak1_i == corr.shape[0]-1:
        peak1_i -=1
    if peak1_j == corr.shape[1]-1:
        peak1_j -= 1
        
        
        
    

    try:
        # the peak and its neighbours: left, right, down, up
        c = corr[peak1_i,   peak1_j]
        cl = corr[peak1_i - 1, peak1_j]
        cr = corr[peak1_i + 1, peak1_j]
        cd = corr[peak1_i,   peak1_j - 1]
        cu = corr[peak1_i,   peak1_j + 1]

        # gaussian fit
        if np.any(np.array([c, cl, cr, cd, cu]) < 0) and subpixel_method == 'gaussian':
            subpixel_method = 'centroid'

        try:
            

            if subpixel_method == 'gaussian':
                subp_peak_position = (peak1_i + ((log(cl) - log(cr)) / (2 * log(cl) - 4 * log(c) + 2 * log(cr))),
                                      peak1_j + ((log(cd) - log(cu)) / (2 * log(cd) - 4 * log(c) + 2 * log(cu))))
                
                if subp_peak_position[0] > corr.shape[0] or subp_peak_position[0] < corr.shape[0]*-1:
                    subpixel_method = 'centroid'
                if subp_peak_position[1]> corr.shape[1]or subp_peak_position[1] < corr.shape[1]*-1:
                    subpixel_method = 'centroid' 
                

            if subpixel_method == 'centroid':
                subp_peak_position = (((peak1_i - 1) * cl + peak1_i * c + (peak1_i + 1) * cr) / (cl + c + cr),
                                      ((peak1_j - 1) * cd + peak1_j * c + (peak1_j + 1) * cu) / (cd + c + cu))
            if subpixel_method == 'parabolic':
                subp_peak_position = (peak1_i + (cl - cr) / (2 * cl - 4 * c + 2 * cr),
                                      peak1_j + (cd - cu) / (2 * cd - 4 * c + 2 * cu))

        except:
            subp_peak_position = default_peak_position

    except IndexError:
        subp_peak_position = default_peak_position
        

    return subp_peak_position[0] - default_peak_position[0], subp_peak_position[1] - default_peak_position[1]







            




