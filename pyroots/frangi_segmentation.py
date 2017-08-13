#! /bin/python3/

"""
Author: @pme1123
Created: Jan 17th, 2017

Frangi Segmentation - combines various functions into a single one for convenience
Frangi Image Loop - For series analysis across directories. 

"""

import os
import pandas as pd
from pyroots import *
from skimage import io, color, filters, morphology, img_as_ubyte, img_as_float
import importlib
import numpy as np
from warnings import warn

def frangi_segmentation(image, 
                        colors,
                        frangi_args, 
                        threshold_args, 
                        contrast_kernel_size='skip',
                        color_args_1='skip',
                        color_args_2='skip', 
                        color_args_3='skip', 
                        neighborhood_args='skip',
                        morphology_args_1='skip', 
                        morphology_args_2='skip', 
                        hollow_args='skip', 
                        fill_gaps_args='skip', 
                        diameter_args='skip', 
                        diameter_bins='skip', 
                        image_name='image', 
                        verbose=False):
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
    contrast_kernel_size : int, str, or None
        Kernel size for `skimage.exposure.equalize_adapthist`. If `int`, then gives the size of the kernel used
        for adaptive contrast enhancement. If `None`, uses default (1/8 shortest image dimension). If `skip`,
        then skips. 
    color_args_1 : dict
        Parameters to pass to `pyroots.color_filter`.
    color_args_2 : dict
        Parameters to pass to `pyroots.color_filter`. Combines with color_args_1
        in an 'and' statement.
    color_args_3 : dict
        Parameters to pass to `pyroots.color_filter`. Combines with color_args_1, 2
        in an 'and' statement.
    neighborhood_args : dict
        Parameters to pass to 'pyroots.neighborhood_filter'. 
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
    # convert band to list for downstream compatibilty, if necessary
    if len(colors) == 3: # it's a color image
        try:
            len(colors['band'])
        except: 
            colors['band'] = [colors['band']]
            pass

        # convert colorspace, if necessary
        if colors['colorspace'].lower() != 'rgb':
            working_image = getattr(color, "rgb2" + colors['colorspace'])(working_image)
        else:
            working_image = working_image.copy()
        
        # select band
        if len(working_image.shape) == 3:  # excludes rgb2gray
            working_image = [img_split(working_image)[i] for i in colors['band']]
        else:
            working_image = [working_image]

        nbands = len(colors['band'])
    
    else:  # it's a black and white image
        if working_image.shape == 3:
            raise ValueError("Color images require color image parameters. Yours are black and white parameters.")
        working_image = [working_image]
        nbands = 1

    if verbose is True:
        print("Color bands selected")
    
    working_image = [img_as_float(i) for i in working_image]
    
    # Contrast enhancement
    try:
        for i in range(nbands):
            temp = exposure.equalize_adapthist(working_image[i], 
                                               kernel_size = contrast_kernel_size)
            working_image[i] = img_as_float(temp)
        if verbose is True:
            print("Contrast enhanced")
    except:
        if contrast_kernel_size is not 'skip':
            warn('Skipping contrast enhancement')
        pass
        
    # invert if necessary
    for i in range(nbands):
        if colors['dark_on_light'][i] is False:
            working_image[i] = 1 - img_as_float(working_image[i])
    
    # Frangi vessel enhancement
    working_image = [filters.frangi(i, **frangi_args) for i in working_image]
    working_image = [1-(i/np.max(i)) for i in working_image]  # rescale and invert --> dark objects for thresholding
    if verbose is True:
        print("Frangi filter complete")
    
    # Threshold to ID candidate objects
    for i in range(nbands):
        working_image[i] = working_image[i] < filters.threshold_local(working_image[i], **threshold_args[i])
    if verbose is True:
        print("Thresholding complete")
    
    # Combine bands
    combined = working_image[0].copy()
    for i in range(1, nbands):
        combined = combined * working_image[i]
    working_image = combined.copy()
    
    # Filter candidate objects by color
    try:
        color1 = color_filter(image, working_image, **color_args_1)  #colorspace, target_band, low, high, percent)
        if verbose is True:
            print("Color filter 1 complete")
    except:
        if color_args_1 is not 'skip':
            warn("Skipping Color Filter 1")
        color1 = np.ones(working_image.shape)  # no filtering      

    try:
        color2 = color_filter(image, working_image, **color_args_2)  # nesting equates to an "and" statement.
        if verbose is True:
            print("Color filter 2 complete")   
    except:
        if color_args_2 is not 'skip':
            warn("Skipping Color Filter 2")
        color2 = np.ones(working_image.shape)  # no filtering
    
    try:
        color3 = color_filter(image, working_image, **color_args_3)  # nesting equates to an "and" statement.
        if verbose is True:
            print("Color filter 3 complete")
    except:
        if color_args_3 is not 'skip':
            warn("Skipping Color Filter 3")
        color3 = np.ones(working_image.shape)  # no filtering
    
    # Combine bands
    working_image = color1 * color2 * color3
    
    # Filter objects by neighborhood colors
    try:
        working_image = neighborhood_filter(image, working_image, **neighborhood_args)
        if verbose is True:
            print("Neighborhood filter complete")
    except:
        if neighborhood_args is not 'skip':
            warn("Skipping neighborhood filter")
        pass
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args_1)
        if verbose is True:
            print("Morphology filter 1 complete")
    except:
        if morphology_args_1 is not 'skip':
            warn("Skipping morphology filter 1")
        pass
    
    # Filter candidate objects by hollowness
    try:  
        working_image = hollow_filter(working_image, **hollow_args)
        if verbose is True:
            print("Hollow filter complete")
    except:
        if hollow_args is not 'skip':
            warn("Skipping hollow filter")
        pass
    
    # Close small gaps and holes in accepted objects
    try:
        working_image = fill_gaps(working_image, **fill_gaps_args)
        if verbose is True:
            print("Gap filling complete")
    except:
        if fill_gaps_args is not 'skip':
            warn("Skipping filling gaps")
        pass
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args_2)
        if verbose is True:
            print("Morphology filter 2 complete")
    except:
        if morphology_args_2 is not 'skip':
            warn("Skipping morphology filter 2")
        pass
        
    # Skeletonize. Now working with a dictionary of objects.
    skel = skeleton_with_distance(working_image)
    if verbose is True:
        print("Skeletonization complete")
    
    # Diameter filter
    try:
        diam = diameter_filter(skel, **diameter_args)
        if verbose is True:
            print("Diameter filter complete")
    except:
        diam = skel.copy()
        if diameter_args is not 'skip':
            warn("Skipping diameter filter")
        pass
    
    # Summarize
    if diameter_bins is None or diameter_bins is 'skip':
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

    if verbose is True:
        print("Done")

    return(out)
