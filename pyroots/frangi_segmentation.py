#! /bin/python3/

"""
Author: @pme1123
Created: Jan 17th, 2017

Frangi Segmentation - combines various functions into a single one for convenience
Frangi Image Loop - For series analysis across directories. 

"""

# TODO: Reminder - don't forget to update these imports!
import os
# from PIL import Image
import pandas as pd
from pyroots.image_manipulation import img_split, fill_gaps
from pyroots.geometry_filters import morphology_filter, hollow_filter, diameter_filter
from pyroots.noise_filters import color_filter
from pyroots.summarize import bin_by_diameter, summarize_geometry
from pyroots.skeletonization import skeleton_with_distance
from skimage import io, color, filters, morphology, img_as_ubyte, img_as_float
import importlib
import numpy as np
from warnings import warn

def frangi_segmentation(image, colors, frangi_args, threshold_args,
                        color_args_1=None, color_args_2=None, color_args_3=None, 
                        morphology_args_1=None, morphology_args_2=None, hollow_args=None, 
                        fill_gaps_args=None, diameter_args=None, diameter_bins=None, 
                        image_name="image"):
    """
    Possible approach to object detection using frangi filters. Selects colorbands for
    analysis, runs frangi filter, thresholds to identify candidate objects, then removes
    spurrious objects by color and morphology characteristics. See frangi_approach.ipynb. 
    
    Unless noted, the dictionaries are called by their respective functions in order.
    
    Parameters
    ----------
    image : ndarray
        RGB image to analyze
    colors : dict
        Parameters for picking the colorspace
    frangi_args : dict
        Parameters to pass to `skimage.filters.frangi`
    threshold_args : dict
        Parameters to pass to `skimage.filters.threshold_adaptive`
    color_args_1 : dict
        Parameters to pass to `pyroots.color_filter`.
    color_args_2 : list
        Parameters to pass to `pyroots.color_filter`. Combines with color_args_1
        in an 'and' statement.
    color_args_3 : list
        Parameters to pass to `pyroots.color_filter`. Combines with color_args_1, 2
        in an 'and' statement.
    morphology_args_1 : dict
        Parameters to pass to `pyroots.morphology_filter`    
    morphology_args_2 : dict
        Parameters to pass to `pyroots.morphology_filter`. Happens after fill_gaps_args in the algorithm.
    hollow_args : dict
        Parameters to pass to `pyroots.hollow_filter`
    fill_gaps_args : dict
        Paramaters to pass to `pyroots.fill_gaps`
    diameter_bins : list
        To pass to `pyroots.bin_by_diameter`
    optimize : bool
        Print images automatically, to facilitate parameter tweaking
    image_name : str
        Identifier of image for summarizing
    
    Returns
    -------
    A dictionary containing:
        1. `"geometry"` summary `pandas.DataFrame`
        2. `"objects"` binary image
        3. `"length"` medial axis image
        4. `"diameter"` medial axis image
 
    """

    working_image = image.copy()
    
    # Pull band from colorspace
    color_conversion = 'rgb2' + colors['colorspace'].lower()
    try:
        working_image = getattr(color, color_conversion)(working_image)
    except:
        pass
    
    working_image = img_split(working_image)[colors['band']]
    
    if colors['invert'] is True:
        working_image = 1 - img_as_float(working_image)
    
    # Frangi vessel enhancement
    working_image = filters.frangi(working_image, **frangi_args)
    working_image = 1-(working_image/working_image.max())  # rescale and invert --> dark objects for thresholding

    # Threshold to ID candidate objects
    working_image = filters.threshold_adaptive(working_image, **threshold_args)
    working_image = ~working_image  # new line for clarity; --> objects = True
    
    # Filter candidate objects by color
    try:
        color1 = color_filter(image, working_image, **color_args_1)  #colorspace, target_band, low, high, percent)
    except:
        if color_args_1 is not None:
            warn("Skipping Color Filter 1")
        color1 = np.ones(working_image.shape)  # no filtering
         
    try:
        color2 = color_filter(image, working_image, **color_args_2)  # nesting equates to an "and" statement.
    except:
        if color_args_2 is not None:
            warn("Skipping Color Filter 2")
        color2 = np.ones(working_image.shape)  # no filtering
    
    try:
        color3 = color_filter(image, working_image, **color_args_3)  # nesting equates to an "and" statement.
    except:
        if color_args_3 is not None:
            warn("Skipping Color Filter 3")
        color3 = np.ones(working_image.shape)  # no filtering
    
    working_image = color1 * color2 * color3
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args_1)
    except:
        if morphology_args_1 is not None:
            warn("Skipping morphology filter 1")
        pass
    
    # Filter candidate objects by hollowness
    try:  
        working_image = hollow_filter(working_image, **hollow_args)
    except:
        if hollow_args is not None:
            warn("Skipping hollow filter")
        pass
    
    # Close small gaps and holes in accepted objects
    try:
        working_image = fill_gaps(working_image, **fill_gaps_args)
    except:
        if fill_gaps_args is not None:
            warn("Skipping filling gaps")
        pass
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args_2)
    except:
        if morphology_args_2 is not None:
            warn("Skipping morphology filter 2")
        pass
        
    # Skeletonize. Now working with a dictionary of objects.
    skel = skeleton_with_distance(working_image)
    
    # Diameter filter
    try:
        diam = diameter_filter(skel, **diameter_args)
    except:
        if diameter_args is not None:
            warn("Skipping diameter filter")
        pass
    
    # Summarize
    if diameter_bins is None:
        summary_df = summarize_geometry(diam['geometry'], image_name)

    else:
        diam_out, summary_df = bin_by_diameter(diam['length'],
                                               diam['diameter'],
                                               diameter_bins,
                                               image_name)
        diam['diameter'] = diam_out
    
    out = {'geometry' : summary_df,
           'objects'  : diam['objects'],
           'length'   : diam['length'],
           'diameter' : diam['diameter']}
    
    return(out)
