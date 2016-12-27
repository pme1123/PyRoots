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
    labels : the output of ndimage.label(), which labels image objects in binary
    images. For skeletons, use maximum distance.
    diameter_image : skeleton with pixel values as diameter, for example.
    percentile : the first classification parameter, the percentile of pixel 
    values at which to make the decision to keep
    value : the second classification parameter, the threshold at which the
    classification of an image object switches
    test_type : Do the percentiles and values indicate minimum thresholds, or
    maximum thresholds? Options are "ceiling" or "floor".
    
    Returns
    -------
    A binary array
    
    See Also
    --------
    numpy.percentile, pyroots.diameter_filter
    
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


def diameter_filter(diameter_image, length_image, objects_image,
            max_diameter=1000, min_diameter=-1, 
            max_percentile=100, min_percentile=None):
    """
    Remove objects based on width thresholds. For example, hyphae usually are 
    < 5um diameter, so objects that are mostly >5um are not hyphae, and 
    'objects' that are composits of debris and hyphae will have parts that are
    > 5um. This function removes both. It also provides the option to provide a
    floor for diameter. Requires scipy.
    
    Parameters
    ----------
    diameter_image : the diameter skeleton array of the thresholded (object) 
        image, for example pyroots.skeleton_with_distance(binary_image)[1]
    length_image : the length skeleton array of the thresholded (object) 
        image, for example pyroots.skeleton_with_distance(binary_image)[0]
    objects_image : Binary array of objects, for example feeding into 
        pyroots.skeleton_with_distance.
    max_percentile : Of all of the skeleton pixels for a single object, if the 
        ceiling is smaller than the percentile, the entire object is deleted. 
        Feeds into numpy.percentile(). Default=100 (effectively no filter).
    max_diameter : Cutoff, where true objects have a narrower (exclusive) 
        diameter. In pixels. Default=1000 (effectively no filter).
    min_percentile : Of all of the skeleton pixels for a single object, if the 
        floor is larger than the percentile, the entire object is deleted. 
        Feeds into numpy.percentile(). Default=None.
    min_diameter : Cutoff, where true objects have a wider (exclusive) diameter. 
        In pixels. Default=-1 (for no filtering).
    
    Returns
    -------
    A list of four objects:
    1. A pandas dataframe of updated mean diameter and total length (in 
        pixels) of each object. 
    2. A filtered skeleton array of pixel diameter 
    3. A filtered skeleton array of pixel length 
    4. An updated list of image objects
    
    See Also
    --------
    pyroots._percentile_filter, scipy.ndimage.label, pandas
    
    """    
#    from pyroots import _percentile_filter
    
    # make sure diameter, length skeletons match objects image
    diameter_image = diameter_image * objects_image
    length_image = length_image * objects_image
    
    #Label objects for indexing
    labels, labels_ls= ndimage.label(diameter_image>0,
               structure = np.ones((3,3)))
      
    #Percentile filters              
    max_perc_filter = _percentile_filter(labels, diameter_image, 
                                         max_percentile, max_diameter,
                                         'ceiling')
                    
    if min_percentile != None:
        min_perc_filter = _percentile_filter(labels, diameter_image,
                                             min_percentile, min_diameter, 
                                             'floor')
        perc_filter = max_perc_filter * min_perc_filter
    else:
        perc_filter = max_perc_filter
    
    # Max, min diameter filter
    max_diam_filter = diameter_image < max_diameter
    min_diam_filter = diameter_image > min_diameter
    
    diam_filter = max_diam_filter * min_diam_filter * perc_filter
    
    new_diam_skeleton = diameter_image * diam_filter
    new_len_skeleton = length_image * diam_filter
    new_labels = labels * diam_filter
    
    # Re-calculate geometry
    width_list = ndimage.mean(new_diam_skeleton, 
                  new_labels, 
                  index=range(new_labels.max()+1)) 
    len_list = ndimage.sum(new_len_skeleton,
               new_labels,
               index = range(new_labels.max()+1))
    
    geom_out = pd.DataFrame({'Length' : len_list,
                             'Diameter' : width_list
                             })
    geom_out = geom_out.drop([0])
    
    labels, labels_ls = ndimage.label(objects_image)
    new_objects = np.in1d(labels, np.unique(new_labels))
    new_objects = np.reshape(new_objects, new_labels.shape) * labels
    return(new_objects, geom_out, new_diam_skeleton, new_len_skeleton)
    
def length_width_filter(binary_image, geometry, threshold=5):
    """
    Remove objects based on length:(average) width ratios from skeletonized images.
    
    Parameters
    ----------
    binary_image : a in input binary image
    geometry : n*2 array of geometry for each object. 
    threshold : Minimum length:width ratio to keep an object. Default = 5.
    
    Returns
    -------
    A list containing two objects. 1) is a binary array with only objects that
    have a large enough length:width threshold. 2) is the updated geometry dataframe.
    
    """    

    geometry = [geometry['Diameter'].values, geometry['Length'].values]
    labels, labels_ls = ndimage.label(binary_image)
    
    if labels_ls + 1 is not len(geometry[0]):
        return("Incompatible Geometry Array and Image")
    
    #Calculate length:width ratios in the geom array and test whether they pass
    ratio = geometry[1] / (geometry[0]+0.00001)
    thresh_test = ratio > threshold
    
    #convert the labels to a boolean taking the value from thresh_test based on
    #the index defined by labels
    out = thresh_test[labels]
    
    #scalar multiply to remove geometry of objects that don't pass the threshold
    geom_thresh = geometry * thresh_test
    its = range(len(geom_thresh))
    
    geom_out = [i for i in its] #placeholder
    for i in its:
        geom_out[i] = np.insert(
            [j for j in geom_thresh[i] if j != 0], #remove rows equal to zero
            0,0) #re-add first row of zeros for compatibilty with ndimage.label
    
    geom_out = pd.DataFrame({'Length' : geom_out[1],
                             'Diameter' : geom_out[0]
                             })
    return(out, geom_out)
