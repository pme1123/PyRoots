# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:10:25 2016

@author: pme

Contents:
- summarize_by_geometry
- bin_by_diameter
"""

import pandas as pd
import numpy as np
from scipy import ndimage

def summarize_geometry(pyroots_geom, image_name):
    """
    Created: 08/11/2016
    
    @author : PME
    
    Summarize the geometry ndarray from pyroots.length_width_filter,  
    pyroots.skeleton_with_distance, or pyroots.diam_binning into a pandas
    DataFrame.
    
    Parameters
    ----------
    pyroots_geom : output geometry pandas.DataFrame from pyroots functions. 
        Columns denote length or diameter. Rows index objects within an image.
    image_name : the name used to identify this image uniquely. Usually the file
        name.
    
    Returns
    -------
    A pandas DataFrame with obvious column headings
    
    See Also
    --------
    pandas.DataFrame
    """
    
    #Add ObjectNumber and DiamWeight columns to calculate
    #total objects and mean diameter, respectively 
    pyroots_geom.insert(1, 'Object', pyroots_geom.index)
    pyroots_geom['DiamWeight'] = pyroots_geom['Length'] * pyroots_geom['Diameter']
    
    #Calculate summary statistics
    summary_df = pd.DataFrame({'ImageName' : pd.Series(image_name), #slicing
                               'NObjects' : pd.Series(pyroots_geom['Object'].max()),
                               'Length' : pd.Series(pyroots_geom['Length'].sum())
                               })
    summary_df['MeanDiam'] = pyroots_geom['DiamWeight'].sum() / summary_df['Length']
    
    return(summary_df)


def bin_by_diameter(length_skeleton, diameter_skeleton, breakpoints):
    """
    Bin objects into diameter classes and calculate the total length of each
    class.
    
    Parameters
    -----
    length_skeleton : non-binary medial axis array showing the length value
    of each pixel in the skeleton along the axis
    diameter_skeleton : non-binary medial axis array showing the diameter value
    of each pixel in the skeleton across the axis
    breakpoints : a list of break points for binning diameter classes, in pixels
    
    Returns
    -------
    A list containing:
    **1)** A data array of length by bin class
    **2)** An image array visualizing the class of each pixel of the medial axis
    
    """
    diameter_skeleton = diameter_skeleton.astype('int64')    
    
    #Create a histogram
    bins = np.bincount(diameter_skeleton.ravel())
    breakpoints.insert(0,0)
    bins_index = range(len(breakpoints))
    
    #Reclassify pixels by bins
    bins_reclass = bins.copy()
    for i in bins_index:
        begin = breakpoints[i]
        bins_reclass[begin:] = bins_index[i]

    
    bins_reclass = bins_reclass[diameter_skeleton]
    
    #calculate length of each diameter class
    bins_length = ndimage.sum(length_skeleton,
                              labels = bins_reclass,
                              index = range(bins_reclass.max()+1)
                              )
    
    #create output
    breakpoints.append(len(bins))
    del breakpoints[0]
    bins_length_out = pd.DataFrame({
        "DiameterClass" : breakpoints,
        "Length" : bins_length
        })  
            
    return(bins_reclass, bins_length_out)
