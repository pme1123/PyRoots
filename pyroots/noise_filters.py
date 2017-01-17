# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:46:19 2016

@author: pme

Contents:
Various methods for filtering and cleaning up candidate objects.
- noise_removal: separates loosely connected objects, removes extremely small objects.
- dirt_removal: removes objects based on size, using statistical methods rather than thresholds.
- grayscale_filter, color_filter: based on object values in individual bands of different colorspaces.
"""

from scipy import ndimage
from skimage import morphology, filters, color, img_as_float  # generally ~5x faster than ndimage
import numpy as np
from pyroots import img_split


#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                            Grayscale and Color Filters, Supporting Functions                             ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def _in_range(image, low, high):
    """
    Rotates and restacks image so `low` = 0. Returns boolean array if rotated image
    pixel value is within (low, high) range.
    
    Parameters
    ----------
    image : ndarray (int, float)
        Grayscale image or band of colorspace
    low : int, float
        Low value
    high : int, float
        High value. Can be smaller than low. Useful for polar spaces like hue in hsv (polar scale)
    
    Returns
    -------
    A boolean ndarray
    """
    img = img_as_float(image)
    
    # rotate so that low = 0. For polar scales (hue of hsv). Arbitrary for 
    img = img - low
    img_a = img + 1
    img[img<0] = img[img<0] + 1  # negative numbers get moved to the top.
    
    new_high = high - low
    if new_high < 0:
        new_high += 1
    
    out = (img >= 0) * (img <= new_high)
    return(out)


def grayscale_filter(image, objects, low, high, percent):
    """
    Determines whether each object in `objects` has values within the range given
    in `image`. Handles polar spaces by automatically 'rotating' and restacking the 
    values such that `low` = 0. Therefore, `high` can be lower than `low`.
    
    Parameters
    ----------
    image : ndarray (int, float)
        Grayscale, including single band of a color image.
    objects : ndarray (bool)
        Potential objects. Binary image.
    low : float
        low value of range. In fraction of colorspace [0:1]
    high : float
        high value of range. In fraction of colorspace [0:1]
    percent : float
        percent of pixels that must be within (low:high). [0, 100].
    
    Returns
    -------
    An updated objects image.
    """
    labels = ndimage.label(objects)[0]
    
    # Calculate area of objects
    binary_area = ndimage.sum(objects, labels=labels, index=range(labels.max()+1))
    
    # Calculate number of pixels in range for each object
    in_range = _in_range(image, low, high)      # flag image pixel values
    in_range_area = ndimage.sum(in_range, labels=labels, index=range(labels.max()+1))
       # Calculate area of in-range pixels for each object
    
    # Do percentage of pixels in range meet threshold?
    test = in_range_area/binary_area >= percent/100
    test[0] = False  # background
    
    # update objects
    out_objects = np.array(test)[labels]
    return(out_objects)

def color_filter(image, objects, colorspace, target_band, low, high, percent):
    """
    Wrapper for `pyroots.grayscale_filter`. Adds functionality to (optionally) convert an rgb image to
    a selected colorspace, and choose a single band from that colorspace. Tests whether `percent` of pixels
    in each object in `objects` fall into (low:high) of `colorspace`[target_band] in `image`. 
    
    Parameters
    ----------
    image : ndarray (int, float)
        RGB image
    objects : ndarray (bool)
        Candidate objects
    colorspace : str
        What colorspace do you want? If not `'rgb'`, must complete `skimage.color.rgb2***`.
    target_band : int
        What band of `colorspace` do you want in [0:2]?
    low : float
        low value of range. In fraction of colorspace [0:1]
    high : float
        high value of range. In fraction of colorspace [0:1]
    percent : float
        percent of pixels that must be within (low:high). [0, 100].
    
    Returns
    -------
    An updated boolean ndarray of candidate objects. objects. 
       
    """
    # convert rgb image if necessary, select band.
    colorband = image.copy()
    if colorspace.lower() is not 'rgb':
        colorband = getattr(color, "rgb2" + colorspace.lower())(colorband)

    colorband = img_split(colorband)[target_band]

    # Filter color
    out = grayscale_filter(colorband, objects, low, high, percent)      ##### ADD PR
    return(out)


#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                           Noise Removal Filter                                           ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################


def noise_removal(img, structure_1="Default", structure_2="Default", median_iterations=2):
    """
    Cleans a binary image by separating loosely connected objects, eliminating
    small objects, and finally smoothing edges of the remaining objects. 
    The method is ``binary_opening``, ``binary_closing``, and two rounds of 
    ``median_filter``.
    
    Parameters
    ----------
    img : array
    	a boolean ndarray
    structure_1 : array
    	structuring element for opening and closing (boolean ndarray). Default
        is manhattan distance = 1 (corners = 0).
    structure_2 : array
    	structuring element for smoothing with a median filter (boolean ndarray). 
    	Default is euclidean distance < sqrt(2).

    Returns
    -------
    A boolean ndarray of the same dimensions as ``img``.
    
    See Also
    --------
    In ``scipy.ndimage``, see ``skimage.morphology.binary_opening``, 
    ``skimage.morphology.binary_closing``, and ``skimage.filters.median``
    
    """

    if structure_1 is "Default":
        ELEMENT_1 = [[0,1,0],
                     [1,1,1],
                     [0,1,0]]
    else:
        ELEMENT_1 = structure_1
    
    if structure_2 is "Default":
        ELEMENT_2 = [[0,1,1,1,0],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [1,1,1,1,1],
                     [0,1,1,1,0]]
    else:
        ELEMENT_2 = structure_2
        
    out = morphology.binary_opening(img, selem=ELEMENT_1)
    out = morphology.binary_closing(out, selem=ELEMENT_1)
    
    i = 0
    while i <= median_iterations:
        out = filters.median(out, selem=ELEMENT_2)
        i += 1

    return(out)
    

#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                       Dirt (small objects) Filter                                        ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def dirt_removal(img, method="gaussian", param=5):
    """
    Removes objects based on size. Uses either a statistical (gaussian)
    cutoff based on the attributes of all objects in the image, or a threshold
    for minimum area (equivalent to ``skimage.morphology.remove_small_objects(img, min_size)``). 
    Requires ``numpy`` and ``scipy``.

    Parameters
    ---------
    img : array
    	boolean ndarray or binary image
    method : str
    	use statistical filtering based on image parameters? Options are 
        ``"gaussian"`` (default) or ``"threshold"``.
    param : float
    	Filtering parameter. For ``method="gaussian"``, ``param`` defines the number
        of standard deviations larger than the median area as the cutoff, above
        which objects are considered 'real'. For ``method="threshold"``, ``param`` 
        identifies the minimum size in pixels. Default = 5
    
    Returns
    --------
    A binary image
    """
    
    labels, labels_ls = ndimage.label(img)
    area = ndimage.sum(img, labels=labels, index=range(labels_ls))
    
    if method is "gaussian": #ID 'real' objects
        area_filt = area > np.median(area) + param*np.std(area)
    elif method is "threshold":
        area_filt = area > param
    else:
        print("method should be 'gaussian' or 'threshold'!")
    
    keep_ID = [i for i, x in enumerate(area_filt) if x == True] #Select labels of 'real' objects
    filt = np.in1d(labels, keep_ID) #reshape to image size
    filt = np.reshape(filt, img.shape)
    return(filt)



