# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 09:46:19 2016

@author: pme

Contents:
- _percentile_filter
- diameter_filter
- length_width_filter

"""

from scipy import ndimage
import pandas as pd
import numpy as np

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
        temp = diameter_image[labels == i+1]  # 0 is background
        temp = np.percentile(temp, percentile)
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
    diameter = diameter_in * objects_in
    length = length_in * objects_in
    
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
                             'Diameter' : objects_diameter
                             })
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
    
    if labels_ls + 1 is not len(geometry_in.index):
        return("Incompatible Geometry Array and Image")
    
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
