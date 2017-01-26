# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:46:19 2016

@author: pme

Contents:
Various functions for removing candidate objects based on their geometry. 
- _percentile_filter: Supports diameter filter
- diameter_filter: Based on diameter along the medial axis
- length_width_filter: Based on length and diameter along the medial axis.
- morphology_filter: Based on properties of convex hulls and equivalent ellilpses
- hollow_filter: Based on medial axis lengths of original and filled objects
"""

from scipy import ndimage
import pandas as pd
import numpy as np
from skimage import morphology, measure
from pyroots.skeletonization import _axis_length

#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                   Diameter Filter, Percentile Filter                                     ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def _percentile_filter(labels, diameter_image, percentile, value, test_type):
    """
    Function to classify (boolean) image objects based the distribution of pixel
    values in the object. Acts as either a maximum or minimum filter.
    
    Parameters
    ----------
    labels : array
        the output of ``ndimage.label``, which labels image objects in binary
        images. For skeletons, use maximum distance rather than manhattan.
    diameter_image : array
        skeleton image with pixel values as diameter and background as 0, for 
        example.
    percentile : float
        the percentile of pixel values at which to make the decision to keep
    value : float
        the threshold at which the classification of an image object switches
    test_type : str
        Do the percentiles and values indicate minimum thresholds, or
        maximum thresholds? Options are "ceiling" or "floor".
    
    Returns
    -------
    A binary array
    
    See Also
    --------
    ``numpy.percentile``, ``pyroots.diameter_filter``
    
    """
    
    # calculate percentile diameter for each object
    out = []
    for i in range(labels.max()):
        temp = np.ma.masked_array(skel['diameter'], labels != i+1)  # 0 is background
        temp = np.percentile(temp.compressed(), percentile)
        out.append(temp)
    
    #select objects that meet diameter criteria at percentile
      # i=0 is background, therefore i+1
    if test_type == "ceiling":
        out = [i+1 for i, x in enumerate(out) if x < value]   
    elif test_type == "floor":
        out = [i+1 for i, x in enumerate(out) if x > value]        
    else:
        print("Test_type should be 'ceiling' or 'floor'!")
    
    #translate this to the labels image
    out = np.in1d(labels, out)  # is each pixel value represented in the 
    # list of objects to keep?
    out = np.reshape(out, labels.shape)  # reshape to image
    
    return(out)


def diameter_filter(skeleton_dictionary,
            max_diameter=1000, min_diameter=-1, 
            max_percentile=100, min_percentile=None, pixel_level = False):
    """
    Remove objects based on width thresholds. For example, hyphae usually are 
    < 5um diameter, so objects that are mostly >5um are not hyphae, and 
    'objects' that are composits of debris and hyphae will have parts that are
    > 5um. This function removes both. It also provides the option to provide a
    floor for diameter. Requires ``scipy``.
    
    Parameters
    ----------
    skeleton_dictionary : dict
        Standard dictionary of objects returned from ``pyroots.skeleton_with_distance``.
        Items are: 
            * "objects", a binary image of objects
            * "length", an ndarray of medial axis, with each pixel representing length
            * "diameter", an ndarray of medial axis, with each pixel representing diameter
            * "geometry", a pandas ``DataFrame`` of total length and average diameter for
            each object.
    max_percentile : float
        Of all of the skeleton pixels for a single object, if the ceiling is smaller 
        than the percentile, the entire object is deleted. Feeds into 
        ``numpy.percentile``. Default=100 (effectively no filter).
    max_diameter : float
        Cutoff, where true objects have a narrower (exclusive) diameter. In pixels.
        Default=1000 (effectively no filter).
    min_percentile : float
        Of all of the skeleton pixels for a single object, if the floor is larger 
        than the percentile, the entire object is deleted. Feeds into 
        ``numpy.percentile``. Default=``None``.
    min_diameter : float
        Cutoff, where true objects have a wider (exclusive) diameter. In pixels. 
        Default=-1 (for no filtering).
    pixel_level : bool
        If true, will remove individual pixels with values > ''max_diameter'' and 
        < ''min_diameter''. 
    
    Returns
    -------
    A list of four objects:
    * An updated list of image objects
    * A filtered skeleton array of pixel diameter 
    * A filtered skeleton array of pixel length 
    * A pandas dataframe of updated mean diameter and total length (in 
        pixels) of each object. 
    
    See Also
    --------
    ``pyroots._percentile_filter``, ``scipy.ndimage.label``, ``pandas``
    
    """    
    diameter_in = skeleton_dictionary["diameter"]
    length_in = skeleton_dictionary["length"]
    objects_in = skeleton_dictionary["objects"]

    # make sure skeletons for diameter and length are updated with objects
    diameter = diameter_in * (objects_in > 0)
    length = length_in * (objects_in > 0)
    
    #Label objects for indexing
    labels, labels_ls= ndimage.label(diameter > 0,  # convert float to boolean
                                     structure = np.ones((3,3))) #square for skeletons
      
    ####  Percentile filters  ####              
    max_perc_filter = _percentile_filter(labels, diameter, 
                                         max_percentile, max_diameter,
                                         'ceiling')
                    
    if min_percentile is not None:
        min_perc_filter = _percentile_filter(labels, diameter,
                                             min_percentile, min_diameter, 
                                             'floor')
        perc_filter = max_perc_filter * min_perc_filter
    
    else:
        perc_filter = max_perc_filter
    
    #### Max, min diameter filter ####
    if pixel_level is True:
        max_diam_filter = diameter < max_diameter
        min_diam_filter = diameter > min_diameter
        diam_filter = max_diam_filter * min_diam_filter * perc_filter
    else:
        diam_filter = perc_filter
    
    #### Update the skeletons ####
    new_diam_skeleton = diameter * diam_filter
    new_len_skeleton = length * diam_filter
    new_labels = labels * diam_filter
    
    #### Update geometry dataframe ####
    # Re-calculate geometry of each object
    objects_diameter = ndimage.mean(new_diam_skeleton, 
                                    new_labels, 
                                    index=range(new_labels.max()+1)) 
    objects_length = ndimage.sum(new_len_skeleton,
                                 new_labels,
                                 index=range(new_labels.max()+1))
    
    # Create a new geometry dataframe
    geom_out = pd.DataFrame({'Length' : objects_length,
                             'Diameter' : objects_diameter})
    geom_out = geom_out[geom_out['Diameter'].notnull()]  # subset only present objects
    
    #### update the objects ####
    labels, labels_ls = ndimage.label(objects_in) # make labels or original objects
    new_objects = np.in1d(labels, np.unique(new_labels)) # only keep labels that were kept
    new_objects = np.reshape(new_objects, new_labels.shape) * labels > 0 # maintain as binary
    
    out = {"objects"  : new_objects,
           "length"   : new_len_skeleton,
           "diameter" : new_diam_skeleton,
           "geometry" : geom_out}
    
    return(out)
    
#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                           Length-Width Filter                                            ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

    
def length_width_filter(skeleton_dictionary, threshold=5):
    """
    Remove objects based on length:(average) width ratios from skeletonized images.
    
    Parameters
    ----------
    skeleton_dictionary : dict
        Standard dictionary of objects returned from ``pyroots.skeleton_with_distance``.
        Items are: 
            * "objects", a binary image of objects
            * "length", an ndarray of medial axis, with each pixel representing length
            * "diameter", an ndarray of medial axis, with each pixel representing diameter
            * "geometry", a pandas ``DataFrame`` of total length and average diameter for
            each object.
    threshold : float
        Minimum length:width ratio to keep an object. Default = 5.
    
    Returns
    -------
    A list containing two objects. 1) is a binary array with only objects that
    have a large enough length:width threshold. 2) is the updated geometry dataframe.
    
    """    
    
    diameter_in = skeleton_dictionary["diameter"]
    length_in = skeleton_dictionary["length"]
    objects_in = skeleton_dictionary["objects"]
    geometry_in = skeleton_dictionary["geometry"]
    
    # convert pandas.DataFrame to list
    geometry = [geometry_in['Diameter'].values, geometry_in['Length'].values]
    
    
    labels, labels_ls = ndimage.label(skeleton_dictionary["objects"])
    
    if labels_ls + 1 != len(geometry_in.index):
        raise("Incompatible Geometry Array and Image: Image has " + str(labels_ls + 1) + " objects. Geometry DataFrame has " + str(len(geometry_in.index)) + " objects.")
    
    # Calculate length:width ratios in the geom array and test whether they pass
    ratio = geometry_in['Length'] / (geometry_in['Diameter']+0.000001)
    
    thresh_test = ratio > threshold
    
    # Update geometry dataframe
    geom_out = geometry_in[thresh_test]
    geom_out.loc[0] = np.array([0,0])  # re-insert empty space index.
    geom_out = geom_out.sort_index()
    
    # Update objects dataframe. Convert the labels to a boolean by determining whether
    # the object number is true for thresh_test
    new_objects = np.array(thresh_test)[labels]  
    
    new_diam_skeleton = diameter_in * new_objects
    new_length_skeleton = length_in * new_objects
    
    out = {"objects"  : new_objects,
           "length"   : new_length_skeleton,
           "diameter" : new_diam_skeleton,
           "geometry" : geom_out}
    
    return(out)


#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                           Morphology Filter                                              ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################

def morphology_filter(image, loose_eccentricity=0, loose_solidity=1, 
                      strict_eccentricity=0, strict_solidity=1, 
                      min_length=None, min_size=None):
    """
    Removes objects based on properties of convex hulls and equivalent
    ellipses, plus size. Defaults are for no filtering. This algorithm is
    moderately fast, but time increases with number of objects due to the
    need for loops. 
    
    Parameters
    ----------
    image : 2D binary array.
        Candidate objects
    loose_eccentricity, loose_solidity : float
        AND filters. Must pass both levels.
    strict_eccentricity, strict_solidity : float
        OR filters. Must pass one of these
    min_length : int
        in pixels, of ellipse with equivalent moments to convex hull
    min_size : int
        in pixels, of area of candidate object.
    
    Returns
    -------
    2D binary array
    
    See Also
    --------
    `skimage.measure.regionprops`, `ndimage.label`
    """
    
    # Label objects and attach regionprops methods for each label
    labels = ndimage.label(image)[0]
    props = measure.regionprops(labels)

    # calculate eccentricity
    eccentricity = [0] + [i.eccentricity for i in props]
    eccentricity = np.array(eccentricity)[labels]  # make an image based on labels

    # calculate solidity
    solidity = [0] + [i.area / i.convex_area for i in props]
    solidity = np.array(solidity)[labels]  # make an image based on labels
    
    # loose and strict filters
    loose = ((solidity < loose_solidity) * (solidity > 0)) * (eccentricity > loose_eccentricity)  # AND
    strict = ((solidity < strict_solidity) * (solidity > 0)) + (eccentricity > strict_eccentricity)  # OR 
    
    # calculate length
    if min_length is None:
        length = np.ones(image.shape)
    else:
        length = [0] + [i.major_axis_length for i in props]
        length = np.array(length)[labels]  # make an image based on labels
        length = length > min_length
    
    # calculate size
    if min_size is None:
        size = np.ones(image.shape)
    else:
        size = [0] + [i.area for i in props]
        size = np.array(size)[labels]  # make an image based on labels
        size = (size > min_size)       # filter
    
    # Combine and exit. Must pass all. 
    out = strict * loose * length * size  # AND
    return(out)

#########################################################################################################################
#########################################################################################################################
#######                                                                                                          ########
#######                                             Hollow Filter                                                ########
#######                                                                                                          ########
#########################################################################################################################
#########################################################################################################################
    
def hollow_filter(image, ratio=1.5, fill_kernel=15, **kwargs):
    """
    For each object, what is the ratio of A to B where:
        A = medial axis length before filling (~= "perimeter" of hollow objects)
        B = medial axis length after filling (= true medial of hollow objects)
    Filters objects based on ratio, which is a ceiling for true objects. Assumes
    true objects are not hollow. 
    
    This is a relatively slow algorithm, and should be performed last (time
    proportional to number of objects due to loops). 
        
    Parameters
    ----------
    image : 2D binary array
        input image. 
    ratio : float
        Maximum of A:B (see above)
    fill_kernel : int
        Radius of disk, in pixels, used to fill objects.
    **kwargs : dict
        passed on to `pyroots.noise_removal`
    
    Returns
    -------
    A 2D binary array
    
    See Also
    --------
    `skimage.morphology.binary_closing`, `pyroots.skeleton_with_distance`, 
    `pyroots.noise_removal`
    """
    
    img = image.copy()
    
    # Labels, object slices
    labels, labels_ls = ndimage.label(img)
    props = measure.regionprops(labels)  # for slicing the image around objects
    
    # kernel
    kernel = morphology.disk(fill_kernel)
    # Smooth the image. Medial axis is highly sensitive to bumps. 
#     skel = pr.noise_removal(img, **kwargs)
#     skel = morphology.skeletonize(skel)  # pull 'length' medial axis of all original objects
    
    test = [0] * (labels_ls + 1)
    for i in range(labels_ls + 1):    
        # Bounds of slice to only the object of interest
        a, b, c, d = props[i-1].bbox
        a = max(a - fill_kernel, 0)  # include a buffer. Stay within bounds of image.
        b = max(b - fill_kernel, 0)
        c = min(c + fill_kernel, img.shape[1])
        d = min(d + fill_kernel, img.shape[0])

        temp_object = labels[a:c, b:d] == i
        
        # compute original medial axis length
        open_medial = morphology.skeletonize(temp_object)
        open_length = _axis_length(open_medial)[1]  # length float only
        
        #close object and compute new axis length
        closed_medial = morphology.binary_closing(temp_object, selem=kernel)
        closed_medial = morphology.skeletonize(closed_medial)
        closed_length = _axis_length(closed_medial)[1]
        
        # Does the ratio pass the threshold?
        test[i] = open_length/closed_length < ratio
    
    # update image
    out = np.array(test)[labels] * labels > 0
    return(out)
