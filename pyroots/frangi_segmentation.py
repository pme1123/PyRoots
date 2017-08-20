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
                        separate_objects=True, 
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
    colors : dict or str
        Parameters for picking the colorspace. See `pyroots.band_selector`. 
    frangi_args : list of dict or dict
        Parameters to pass to `skimage.filters.frangi`
    threshold_args : list of dict or dict
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

    # Pull band from colorspace
    working_image = band_selector(image, colors)  # expects dictionary (lazy coding)
    nbands = len(working_image)
    if verbose is True:
        print("Color bands selected")
    
    ## Count nubmer of dictionaries in threshold_args and frangi_args. Should equal number of bands. Convert to list if necessary
    try:
        len(threshold_args[0])
    except:
        threshold_args = [threshold_args]
        if nbands != len(threshold_args):
            raise ValueError(
                """Number of dictionaries in `threshold_args` doesn't
                equal the number of bands in `colors['band']`!"""
            )
        pass 
    
    try:
        len(frangi_args[0])
    except:
        frangi_args = [frangi_args]
        if nbands != len(frangi_args):
            raise ValueError(
                """Number of dictionaries in `frangi_args` doesn't 
                equal the number of bands in `colors['band']`!"""
            )
        pass    
    
    working_image = [img_as_float(i) for i in working_image]
    
    # Contrast enhancement
    try:
        for i in range(nbands):
            temp = exposure.equalize_adapthist(working_image[i], 
                                               kernel_size = contrast_kernel_size)
            working_image[i] = img_as_float(temp)
        if verbose:
            print("Contrast enhanced")
    except:
        if contrast_kernel_size is not 'skip':
            warn('Skipping contrast enhancement')
        pass
        
    # invert if necessary
    for i in range(nbands):
        if not colors['dark_on_light'][i]:
            working_image[i] = 1 - working_image[i]
    
    # Detect edges for separating objects
    edges = [np.ones_like(i)] * nbands    # all True
    if separate_objects:
        for i in range(nbands):
            temp = filters.gaussian(working_image[i])
            temp = filters.scharr(temp)
            temp = temp > filters.threshold_otsu(temp)
            edges[i] = morphology.skeletonize(temp)
            
#        edges = [filters.gaussian(i) for i in working_image]
#        edges = [filters.scharr(i) for i in edges]
#        edges = [i > filters.threshold_otsu(i) for i in edges]
#        edges = [morphology.skeletonize(i) for i in edges]
        if verbose:
            print("Edges found")
    
    # Frangi vessel enhancement
    for i in range(nbands):
        temp = filters.frangi(working_image[i], **frangi_args[i])
        temp = 1 - temp/np.max(temp)
        temp = temp < filters.threshold_local(temp, **threshold_args[i])
        working_image[i] = temp.copy()
    
    frangi = working_image.copy()
    if verbose:
        print("Frangi filter, threshold complete")
    
    
    # Combine bands, separate objects
    combined = working_image[0] * ~edges[0]
    for i in range(1, nbands):
        combined = combined * working_image[i] * ~edges[i]
    working_image = combined.copy()
    
    # Filter candidate objects by color
    try:
        color1 = color_filter(image, working_image, **color_args_1)  #colorspace, target_band, low, high, percent)
        if verbose:
            print("Color filter 1 complete")
    except:
        if color_args_1 is not 'skip':
            warn("Skipping Color Filter 1")
        color1 = np.ones(working_image.shape)  # no filtering      

    try:
        color2 = color_filter(image, working_image, **color_args_2)  # nesting equates to an "and" statement.
        if verbose:
            print("Color filter 2 complete")   
    except:
        if color_args_2 is not 'skip':
            warn("Skipping Color Filter 2")
        color2 = np.ones(working_image.shape)  # no filtering
    
    try:
        color3 = color_filter(image, working_image, **color_args_3)  # nesting equates to an "and" statement.
        if verbose:
            print("Color filter 3 complete")
    except:
        if color_args_3 is not 'skip':
            warn("Skipping Color Filter 3")
        color3 = np.ones(working_image.shape)  # no filtering
    
    # Combine bands
    working_image = color1 * color2 * color3
    del color1
    del color2
    del color3
    
    # Re-expand to area
    if separate_objects:
    
        # find edges removed
        temp = [frangi[i] * edges[i] for i in range(nbands)]
        rm_edges = temp[0].copy()
        for i in range(1, nbands):
            rm_edges = rm_edges * temp[i]
        
        # filter by color per criteria above
        try:    color1 = color_filter(image, rm_edges, **color_args_1)
        except: color1 = np.ones(rm_edges.shape)
        try:    color2 = color_filter(image, rm_edges, **color_args_2)
        except: color2 = np.ones(rm_edges.shape)
        try:    color3 = color_filter(image, rm_edges, **color_args_3)
        except: color3 = np.ones(rm_edges.shape)
        
        # Combine color filters
        expanded = color1 * color2 * color3
    else:
        expanded = np.zeros(colorfilt.shape) == 1  # evaluate to false
    
    
    working_image = expanded ^ working_image  # bitwise or
    
    try:    # remove little objects (for computational efficiency)
        working_image = morphology.remove_small_objects(
            working_image, 
            min_size=morphology_args_1['min_size']
        )
    except:
        pass
    if verbose:
        print("Edges re-added")
        
    
    # Filter objects by neighborhood colors
    try:
        working_image = neighborhood_filter(image, working_image, **neighborhood_args)
        if verbose:
            print("Neighborhood filter complete")
    except:
        if neighborhood_args is not 'skip':
            warn("Skipping neighborhood filter")
        pass
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args_1)
        if verbose:
            print("Morphology filter 1 complete")
    except:
        if morphology_args_1 is not 'skip':
            warn("Skipping morphology filter 1")
        pass
    
    # Filter candidate objects by hollowness
    if hollow_args is not 'skip':  
        working_image = morphology.remove_small_holes(working_image, min_size=10)
        try:
            if np.sum(temp) > 0:
                working_image = hollow_filter(temp, **hollow_args)
            if verbose:
                print("Hollow filter complete")
        except:
            warn("Skipping hollow filter")
            pass
    
    # Close small gaps and holes in accepted objects
    try:
        working_image = fill_gaps(working_image, **fill_gaps_args)
        if verbose:
            print("Gap filling complete")
    except:
        if fill_gaps_args is not 'skip':
            warn("Skipping filling gaps")
        pass
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args_2)
        if verbose:
            print("Morphology filter 2 complete")
    except:
        if morphology_args_2 is not 'skip':
            warn("Skipping morphology filter 2")
        pass
        
    # Skeletonize. Now working with a dictionary of objects.
    skel = skeleton_with_distance(working_image)
    if verbose:
        print("Skeletonization complete")
    
    # Diameter filter
    try:
        diam = diameter_filter(skel, **diameter_args)
        if verbose:
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
