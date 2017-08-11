#! /bin/python3/

"""
Author: @pme1123
Created: August 6th, 2017


Contents:
neighborhod_filter - Filters candidate objects based on pixels near them
"""

from scipy import ndimage
import numpy as np
from skimage import img_as_float, measure, morphology, color
from pyroots.image_manipulation import img_split

def neighborhood_filter(image, objects, max_diff=0.1, gap=4, neighborhood_depth=4, colorspace='rgb', band=2):
    """
    Calculate difference between values on either side of a long, skinny object.
    
    For pyroots, the application is differentiating hyphae or roots from the edges
    of particles. These edges sometimes pass through other filters. True objects 
    (roots and hyphae) should have more or less the same value on either side. Edges
    of larger objects, in comparision, should have a higher value on one side than the other.
    
    This function compares the values on the left and right sides, and upper and lower sides,
    of candidate objects in a grayscale image or band. Based on this difference, the candidate
    object is flagged as real or spurrious.
    
    Parameters
    ----------
    image : array
        1-band, grayscale image, or RGB color image. Converted to float automatically. 
    objects : array
        binary array of candidate objects.
    max_diff : float
        Maximum difference between values in `image` on each side of the candidate objects 
        in `objects`. The magnitude of this value varies with the `colorspace` chosen. For `'rgb'`, 
        the range is [0, 1]. 
    gap : int
        Number of pixels *beyond* each object to start measuring the neighborhood. The width
        of region between the object and the neighborhood. Useful for objects that may not fully
        capture the true object underneath. Default = 4.
    neighborhood_depth : int
        Number of pixels deep that the neighborhood should be. In intervals of 2. Default = 4.
    colorspace : float
        For accessing other colorspaces than RGB. Used to convert a color image to HSV, LAB, etc.
        See `skimage.color`. Ignored if given a 1-band image.
    band : int [0,2]
        Band index for colorspace. Ex. in RGB R=0, G=1, B=2. Ignored if `image` is 1-band. 
        
    Returns
    -------
    A binary array of filtered objects
        
    
    """
    if len(image.shape) == 3:
        if colorspace.lower() != 'rgb':
            image = getattr(color, "rgb2" + colorspace)(image)
        image = img_split(image)[band]
    
    image = img_as_float(image)
    its = int((neighborhood_depth+2)/2)
    gap = int(gap)
    total_dilation = 2*its
    dims = image.shape

    # neighborhood expansion kernels
    kernel_ls = [np.array([[0, 0, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 0, 0]]),      # left
                 np.array([[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [0, 0, 0, 0, 0]]),      # right
                 np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0]]),  # up
                 np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])   # down
                ]

    labels, labels_ls = ndimage.label(objects)
    props = measure.regionprops(labels)

    decision_ls = [False]
    for i in range(1, labels_ls+1):
        ###############
        #### Slice ####
        ###############
        # Bounds of slice to only the object of interest
        # include a gap. Stay within bounds of image.
        a, b, c, d = props[i-1].bbox
        a = max(a - total_dilation, 0)  
        b = max(b - total_dilation, 0)
        c = min(c + total_dilation, dims[1])
        d = min(d + total_dilation, dims[0])
        
        # slice
        obj_slice = labels[a:c, b:d] == i
        img_slice = image[a:c, b:d]
        
        ########################
        ### Local expansion ####
        ########################
        expanded = ~morphology.binary_dilation(obj_slice, morphology.disk(gap))

        nb_ls = []
        median = []
        for k in range(4):
            t = obj_slice.copy()
            for i in range(its):
                t = ndimage.convolve(t, kernel_ls[k])
            nb_ls.append(t * expanded)
            
        ###############################
        #### Select largest object ####
        ###############################
            nb_labels, nb_labels_ls = ndimage.label(nb_ls[k])
            nb_areas = [0] + [i['area'] for i in measure.regionprops(nb_labels)]  # regionprops skips index 0, annoyingly
            if len(nb_areas) == 1:
                nb_areas = nb_areas + [0]
            nb_areas = nb_areas == np.max(nb_areas)  # sometimes (rarely) more than one subregion will have the same (max) area.
            nb_ls[k] = nb_areas[nb_labels]
            
        ##############################################
        #### Find median values of largest object ####
        ##############################################
            masked = np.ma.masked_array(img_slice, ~nb_ls[k]).compressed()
            median.append(np.median(masked))

        ###############################################
        #### Calc difference (left-right, up-down) ####
        ###############################################
        diffs = [np.abs(median[0] - median[1]),
                 np.abs(median[2]- median[3])]
        diffs = [i > max_diff for i in diffs] 
        
        ###################################
        #### Test if exceeds threshold ####
        ###################################
        if sum(diffs) > 0:
            decision_ls.append(False)
        else:
            decision_ls.append(True)
    
    out = np.array(decision_ls)[labels]
    
    return(out)
