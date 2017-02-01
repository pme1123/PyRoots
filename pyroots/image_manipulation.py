"""
Created on Fri May  6 14:55:54 2016

@author: pme


Contents:
- img_split
- circle_mask
- calc_exposure_correction
- _arrays_mean
- _arrays_var
- register_colors
"""

import numpy as np
from skimage import draw, morphology, filters, exposure, img_as_float, img_as_ubyte
from multiprocessing import Pool  
from multiprocessing.dummy import Pool as ThreadPool
import cv2



#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                            Equalize Exposure                                             ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################


def equalize_exposure(image, iterations=1, kernel_size=None, min_object_size=500, dark_objects=True, stretch=False):
    """
    Filter a grayscale image with uneven brightness across it, such as you might see in a microscope image.
    Removes large objects using adaptive thresholding based on `min_object_size`, then calculates the mean
    in a circular neighborhood of diameter `kernel_size`. Smooths this mean, then subtracts the background
    variation from the original image (including large objects). Run twice of best results, though once 
    should give satisfactory results. As a bonus, this often enhances white balance and colors.
    
    For color images, run on each band separately and then combine into a [dim_x, dim_y, 3] numpy array.
    When run on color images with `stretch=True`, this function improves white balance and colors.
    
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

    # Correct Exposure x times
    i = 0
    while i < iterations:
        # Global mean
        img_mean = np.ma.masked_array(img, mask=objects).mean()
        
        # global means
        local_means = filters.rank.mean(img, selem=kernel, mask=~objects)
        local_means = filters.gaussian(local_means, kernel_size)
        
        # Correct Image
        img += (img_mean - local_means)
        img[img>1] = 1  # for compatibilty with img_as_float
        img[img<0] = 0  # for compatibilty with img_as_float
        i += 1

    out = img_as_float(img)
    
    return(out)
    
    
#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                             Calculate Exposure Correction from Images List                               ########
#######                                        and supporting functions                                          ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def _arrays_mean(array_list):
    """
    Calculate the mean of each pixel [i, j, k] in a list of arrays.
    
    Parameters
    ----------
    array_list : list of ndarray
        Must have the same shape. 
    
    Returns
    -------
    ndarray
    """
    dims = array_list[0].shape[2]
    out = np.zeros(array_list[0].shape)
    var_out = out.copy()
     
#    i = 1
    for i in range(dims):
        temp = [j[:, :, i] for j in array_list]
        
        # calculate mean
        means_out = np.zeros(temp[0].shape)
        for k in temp:
            means_out += k  # sum
        
        out[:, :, i] = means_out / len(array_list)  # mean
    
    return(out)
  
def _arrays_var(array_list, mean_img): 
    """
    Calculate the variance of each pixel [i, j, k] in a list of arrays.
    
    Parameters
    ----------
    array_list : list of ndarray
        Must all be same shape.
    mean_img : ndarray
        Output from `pyroots._arrays_mean` or equivalent. 
        
    Returns
    -------
    ndarray
    """ 
    dims = array_list[0].shape[2]
    out = np.zeros(array_list[0].shape)

    for i in range(dims):
        temp = [j[:, :, i] for j in array_list]
        mean = mean_img[:, :, i]
        var_temp = [(k - mean)**2 for k in temp]  # squared error
        
        # calculate mean
        var_out = np.zeros(temp[0].shape)
        for m in var_temp:
            var_out += m  # sum squared error
        
        out[:, :, i] = var_out / (len(array_list)-1)  # variance = mean squared error
    
    return(out)    

def calc_exposure_correction(image_list, smooth_iterations=1, stretch=False, return_variance=False, threads=1):
    """
    Convenience function a common correction to R, G, and B bands from microscope images
    that have systematic differences in brightness. The common correction
    is the average correction of those in image_list. This should deliver a sufficient 
    exposure correction for most images in a dataset without calling `pyroots.equalize_exposure`
    on each image (which would be extremely slow). 
    
    Images in image_list should have relatively few objects that are evenly dispersed,
    and have no large objects. They also should be representative of the color tones of the
    image dataset.
    
    Calls `pyroots.equalize_exposure`.
    
    Parameters
    ----------
    image_list : list of ndarray
        List of images (probably, but not necessarily, rgb) that have unequal exposure. 
    smooth_iterations : int
        How many times do you want to run the `pyroots.equalize_exposure` algorithm on 
        the images? Defaults to `1`.
    stretch : bool
        Rescale band intensities to fill colorspace, for aesthetics? Gives inconsistent
        behavior depending on the individual image; not recommended for analysis purposes. 
        Defaults to `False`.
    return_variance : bool
        Do you also want to calculate the variance of each pixel in the correction image? 
        For diagnostic purposes. Defaults to `False`.
    threads : int
        For processing equalize_exposure in parallel. How many threads do you want to run?
        Speeds up calculation dramatically. See `multiprocessing`. 
    
    Returns
    -------
    If `return_variance=False`, an `ndarray` of the mean value correction for each band.
    
    If `return_variance=True`, a list of two `ndarray`s. The first is the mean value, 
    the second is the variance.
    
    See Also
    --------
    `pyroots.equalize_exposure`, `multiprocessing.dummy.Pool`
    
    """
    
    def _core_fn(image):
        new = img_split(img_as_float(image))
        orig = [i.copy() for i in new]

        i = 0
        while i < smooth_iterations:
            new = [equalize_exposure(i, stretch=stretch) for i in new]  # list
            i += 1
        
        diff = [new[i] - orig[i] for i in range(3)]
        out = np.zeros(image.shape)
        for i in range(image.shape[2]):
            out[:, :, i] = diff[i]
        
        return(out)
        
    #init multiprocessing
    if threads > len(image_list):
        threads = len(image_list)
    
    thread_pool = ThreadPool(threads)
    
    equalized_ = thread_pool.map(_core_fn, image_list)  # run core function
    thread_pool.close()
    thread_pool.join()
    
    out = _arrays_mean(equalized_)  # calculate mean
    
    if return_variance is True:  # calculate variance
        if len(image_list) > 1:
            var = _arrays_var(equalized_, out)
        else:
            var = np.zeros(out.shape)
        
        out = [out, var]

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


#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                             Register Colors                                              ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def register_colors(image, template=1):
    """
    Fix color abberation in images by calculating and applying an affine
    transformation. Color abberation is a result of uneven refraction of light
    of different wavelengths. It shows up as systematic blue and red edges in 
    high-contrast areas.
    
    This should be done before other processing to minimize 
    artifacts.
    
    Parameters
    ----------
    image : ndarray
        3- or 4-channel image, probably RGB.
    template : int
        Band to which to align the other bands. Usually G in RGB. 
    
    Returns
    -------
    An ndarray of `image.size`
    
    Notes
    -----
    Uses `skimage.filters.scharr` to find edges in each band, then finds and
    applies an affine transformation to register the images using 
    `cv2.estimateRigidTransform(fullAffine=False)` and `cv2.warpAffine`.
    """
    
    #find dimensions
    height, width, depth = image.shape
    
    #define bands to analyze
    analyze = []
    for i in range(depth):
        if i != template:
            analyze.append(i)
    
    # Extract bands, find edges
    bands = img_split(image)
    edges = [img_as_ubyte(filters.scharr(i)) for i in bands]

    #make output image
    out = np.zeros((height, width, depth), dtype=np.uint8)
    out[:, :, template] = bands[template]
    
    for i in analyze:
        # calculate transformation
        warp_matrix = cv2.estimateRigidTransform(edges[template],
                                                 edges[i],
                                                 fullAffine=True)
        # transform
        aligned = cv2.warpAffine(bands[i], 
                                 warp_matrix, 
                                 (width, height), 
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,  # otherwise the transformation goes the wrong way
                                 borderMode = cv2.BORDER_CONSTANT)
        
        # add to color image                             
        out[:, :, i] = aligned
    
    return(out)
