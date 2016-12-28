# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:46:19 2016

@author: pme

Contents:
- noise_removal
- dirt_removal
"""

from scipy import ndimage
import numpy as np


def noise_removal(img, structure_1="Default", structure_2="Default"):
    """
    Cleans a binary image by separating loosely connected objects, eliminating
    small objects, and finally smoothing edges of the remaining objects. 
    The method is ``binary_opening``, ``binary_closing``, and two rounds of 
    ``median_filter``. Requires ``scipy``.
    
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
    In ``scipy.ndimage``, see ``binary_opening``, ``binary_closing``, and ``median_filter``
    
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
        
    img_open = ndimage.binary_opening(img, structure=ELEMENT_1)
    img_close = ndimage.binary_closing(img_open, structure=ELEMENT_1)
    img_med1 = ndimage.median_filter(img_close, footprint=ELEMENT_2)
    img_med2 = ndimage.median_filter(img_med1, footprint=ELEMENT_2)
    return(img_med2)

def dirt_removal(img, method="gaussian", param=5):
    """
    Removes objects based on size. Uses either a statistical (gaussian)
    cutoff based on the attributes of all objects in the image, or a threshold
    for minimum area. Requires ``numpy`` and ``scipy``.

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

