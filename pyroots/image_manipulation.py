"""
Created on Fri May  6 14:55:54 2016

@author: pme


Contents:
- img_split
- circle_mask

"""

import numpy as np
from skimage import draw, morphology, filters, exposure, img_as_float


#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                            Equalize Exposure                                             ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################


def equalize_exposure(image, kernel_size=None, min_object_size=500, dark_objects=True, stretch=False):
    """
    Filter a grayscale image with uneven brightness across it, such as you might see in a microscope image.
    Removes large objects using adaptive thresholding based on `min_object_size`, then calculates the mean
    in a circular neighborhood of diameter `kernel_size`. Smooths this mean, then subtracts the background
    variation from the original image (including large objects). Run twice of best results, though once 
    should give satisfactory results. As a bonus, this often enhances white balance and colors.
    
    For color images, run on each band separately and then combine into a [dim_x, dim_y, 3] numpy array.
    When run on color images, this function improves white balance and colors.
    
    Essential for filtering candidate objects by color. Slow; could be optimized with `opencv_python`, though
    this function doesn't support masking when calculating means.
    
    Parameters
    ----------
    image : ndarray (float, int)
        Grayscale image or band.
    kernel_size : int
        Passes to `skimage.morphology.disk` to create a kernel of diameter kernel_size in pixels. If `None`,
        defaults to `max(image.shape)/10`.
    min_object_size : int
        Passes to `skimage.morphology.remove_small_holes`. Area of objects to ignore when averaging
        background values, in pixels.
    dark_objects : bool
        Are objects dark against a light background?
    stretch : bool
        Stretch values to cover entire colorspace? Enhances colors. Largely aesthetic. Not recommended
        for batch analyses.
        
    Returns
    -------
    An ndarray of type float [0:1].
    
    See Also
    --------
    `skimage.filters.rank.mean`
    """
    
    # Housekeeping
    img = img_as_float(image.copy())
    
    if stretch is True:
        img = img/img.max()
    
    if dark_objects is False:
        img = 1-img  # invert
        
    img_in = img.copy()  # for use later
    
    if kernel_size is None:
        kernel_size = np.int(max(image.shape[0], image.shape[1])/10)
    
    # mean filter kernel
    kernel = morphology.disk(int(kernel_size/2))
    
    # identify objects to ignore
    if kernel_size % 2 is 0:
        block_size = kernel_size + 1
    else:
        block_size = kernel_size
    
    objects = ~filters.threshold_adaptive(img, block_size, offset = 0.01*img.max())
    objects = morphology.remove_small_objects(objects, min_size = min_object_size)
    
    # Global mean
    img_mean = np.ma.masked_array(img, mask = objects).mean()

    # Local means
    local_means = filters.rank.mean(img, selem=kernel, mask=~objects)
    local_means = filters.gaussian(local_means, kernel_size)
    
    #correction
    out = img_in + (img_mean - local_means)
    out = img_as_float(out)
    
    return(out)



#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                             Image Splitter                                               ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def img_split(img):

	"""
	Split a multispectral image into its bands. 
	
	Parameters
	----------
	img : array
		A 3- or 4-band image or array

	Returns
	-------
	Returns a list where each section is a separate band
	in the array. ex. RGB image returns [R, G, B].

	"""
	bands = img.shape[2]
	if bands is 1:
		return "Image already is 1D. Why would you split it?"

	band1 = img[:, :, 0]
	band2 = img[:, :, 1]
	band3 = img[:, :, 2]
	if bands is 4:
		band4 = img[:, :, 4]
		return(band1, band2, band3, band4)
	return(band1, band2, band3)


#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                             Elliptical Mask                                              ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def ellipse_mask(img, percentage_x=100, percentage_y=100, offset_x=0, offset_y=0, rotation=0):
    """
    Convenience wrapper for ``skimage.draw.ellipse``. Draws an ellipse at (``center`` + 
    ``offset``) of ``img`` with major and minor axes as a percentage of ``img.shape``. 
    
    Parameters
    ----------
    img : array
        Image array that you want to draw a mask on
    
    percentage_x : float
        What percentage of the x-dimension of ``img`` do you want the x-axis to cover? Can
        be greater than zero
    percentage_y : float
    offset_x : int
        From the middle, how many pixels in the x direction do you want the center of the
        ellipse to be? Positive is left.
    offset_y : int
        Positive is up.
    rotation : float
        In [-pi, pi]. See ``skimage.draw.ellipse''
    
    Returns
    -------
    A binary array with pixels in an ellipse.
    
    References
    ----------
    See source and examples for ``skimage.draw.ellipse``
    
    """
    
    x_rad = np.floor((img.shape[0]/2) * (percentage_x/100))
    y_rad = np.floor((img.shape[1]/2) * (percentage_y/100))
    
    x_center = img.shape[0]//2 + offset_x
    y_center = img.shape[1]//2 - offset_y
    
    mask = np.zeros(img.shape)
    [x, y] = draw.ellipse(y_center, x_center, y_rad, x_rad, shape = img.shape)
    mask[x, y] = 1
    
    return(mask)
