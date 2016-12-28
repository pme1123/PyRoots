"""
Created 08/12/2016 

@author: pme

Contents:
- pyroots_analysis
- image_loop

TODO: Test multi_image_loop
"""

import os
from PIL import Image
import pandas as pd
import pyroots as pr
from skimage import io, color, filters

def pyroots_analysis(image, image_name, colorspace, analysis_bands,
                     threshold_params, mask_params, filtering_params, 
                     light_on_dark = False, diameter_bins = None,
                     optimize = False):
    """
    Full analysis of an image for length of objects based on thresholding. 
    Performs the following steps:
    1. Colorspace conversion
    2. Subsetting analysis bands
    3. Adaptive thresholding
    4. (optional: Edit source) combining thresholded bands
    5. Filter objects by size, length:width ratio, and diameter
    6. Summarizing final result by diameter class or the entire image
    Note that parameters are tricky here. This is meant to be run with a series of
    dictionaries where items have specific names. See individual functions, and
    the example parameters, analysis_parameters.py.
    
    This function is somewhat flexible, but you should edit/re-write it to suit
    your own needs.
    
    Parameters
    ----------
    image : An RGB image for analysis
    image_name : What do you want to call your image?
    colorspace : The colorspace to run the analysis in (ex. RGB, LAB, HSV). String.
    analysis_bands : A list of bands in the colorspace to threshold. ex for HSV, 
        "H" = 0, "S" = 1, "V" = 2. List of integers.
    light_on_dark : Are the objects dark objects on a light background? Default = False.
    threshold_params : Options for adaptive thresholding. See 
        skimage.filters.threshold_adaptive(). Dictionary. 
    mask_params : List of two dictionaries:
    	1. A boolean of what type of mask to use
    	2. A list of numbers specifying the shape of the mask.
    filtering_params : List of two dictionaries:
        1. A size filter threshold for passing to pyroots.dirt_removal(), and a
            length:width value for passing to pyroots.length_width_filter().
        2. Arguments for passing to pyroots.diameter_filter().
    diameter_bins : List of bin cutoffs for summarizing object length by diameter
        class. Default = None.
    optimize : Flag for whether to run the function in a more 'interactive' mode.
        This causes the function to print images at each step, for the purpose
        of tweaking parameters. Default = False.
    
    Returns
    -------
    In optimize mode, a series of plots showing output from individual steps of the
    analysis, plus a pandas dataframe of geometry.
    Otherwise, a list of four objects: 1) geometry dataframe; 2) analyzed objects;
    3) skeleton pixel lengths; 4) skeleton pixel diameters.
    
    See Also
    --------
    analysis_parameters.py
    
    """
    
    ###########################################
    ####  convert colorspace and theshold  ####
    ###########################################
    if colorspace.lower() is not "rgb":
        image = getattr(color, 'rgb2' + colorspace.lower())(image)
    bands = pr.img_split(image)
    
    analysis_band = [bands[n] for n in analysis_bands]
    
    threshold = [filters.threshold_adaptive(n, **threshold_params)
                 for n in analysis_band
                 ]

    if light_on_dark is False:
        threshold = [~n for n in threshold]
        
    
    #############################################################
    #### Insert equation here for combining binary threshold ####
    ####              images of multiple bands               ####
    #############################################################
#    threshold = threshold[1] * threshold[2]  # both g and b of rgb colorspace
    
    # ensure the image is an array and not a list
    threshold = threshold[0]
    
    ##############################################################
    ####   Apply mask to only analyze the interesting part    ####
    ####                   of the image?                      ####
    ##############################################################
        
    if mask[0]['form'] is "ellipse":
        img_mask = pr.circle_mask(threshold, **mask[1])
    threshold = threshold * img_mask  # apply the mask
    
    ##############################################################
    ####  Remove noise, smooth, and clear extraneous objects  ####
    ##############################################################
    no_dirt = pr.dirt_removal(threshold, **filtering_params[0])
    
    raw_objects = pr.noise_removal(no_dirt, 
                                **filtering_params[1])  # can change the stringency of noise
        # noise removal and smoothing by specifying different structure elements
        # see source code (%psource pyroots.noise_removal in ipython)
    
    # medial axis skeleton plus geometry calculations
    raw_length, raw_diam, raw_geom = pr.skeleton_with_distance(raw_objects)
    
    # filter objects by length:width ratio, and update objects
    lw_objects, lw_geom = pr.length_width_filter(raw_objects,
                                                 raw_geom,
                                                 **filtering_params[2])
    
    # filter by diameter
    objects_img, geom, diameter_img, length_img = pr.diameter_filter(raw_diam, 
                                                                  raw_length, 
                                                                  lw_objects,
                                                                  **filtering_params[3])

    ##############################################################
    #### Summarize the length and diameter data for the image ####
    ##############################################################
   
    if diameter_bins is None:
        summary_df = pr.summarize_geometry(geom, image_name)
        
    else:
        diam_out, summary_df = pr.bin_by_diameter(length_img,
                                                   diameter_img,
                                                   diameter_bins)
        summary_df.insert(0, "ImageName", image_name, allow_duplicates=True)
    
    ################################################################
    #### Draw images for setting parameters and troubleshooting ####
    ################################################################
    if optimize is True:
        pr.multi_image_plot([image, bands[0], bands[1], bands[2]],
                         	["Input", "Band 1", "Band 2", "Band 3"])
        pr.multi_image_plot([threshold, no_dirt, raw_objects, lw_objects],
                         	["Threshold", "Small Objects Removal", "Smoothing",
                             "Length Width Filter"])
        pr.multi_image_plot([objects_img, diameter_img, length_img],
                             ["Diameter Filter", "Diameter Skeleton",
                              "Length Skeleton"])
        return(summary_df)
    
    else:
        return(summary_df, objects_img, length_img, diameter_img)

def image_loop(root_directory, image_extension, full_path_to_params, save_images = False):
    """
    Reference function to loop through images in a directory. As it is written, it returns
    a dataframe from "pyroots_analysis" and also writes images showing the objects analyzed.
    """
    source(path_to_params) 
    #Make an output directory for the analyzed images and data. Requires os.
    if save_images is True:
        if not os.path.exists(root_directory + os.sep + "Pyroots Analyzed"):
            os.mkdir(root_directory + os.sep + "Pyroots Analyzed")
    
    #Make a placeholder DataFrame for the output data. Requires pandas.
    if params_dict['diameter_bins'] is None:
        img_df = pd.DataFrame(columns=("ImageName", "Length", "NObjects", "MeanDiam"))    
    else:
        img_df = pd.DataFrame(columns=("ImageName", "DiameterClass", "Length"))
    
    #Begin looping
    for subdir, dirs, files in os.walk(root_directory):
        for file_in in files:
            
            #criteria for doing something
            if file_in.endswith(image_extension) and not subdir.endswith("Pyroots Analyzed"):
                filepath = subdir + os.sep + file_in  # what's the image called and where is it?
    
                #Create filein name
                filein = "DIR" + subdir[len(root_directory):] + os.sep + file_in
                #Import image
                img = io.imread(filepath)[:,:,0:3]  # load image

                #Run through the analysis. Requires pyroots.pyroots_analysis reference function.
                temp_df, objects, length_skel, dist_skel = pr.pyroots_analysis(
                    image = img,
                    image_name = filepath,
                    colorspace = colorspace,
                    analysis_bands = analysis_bands,
                    threshold_params = threshold_params,
                    mask_params = mask_params,
                    filtering_params = filtering_params,
                    light_on_dark = light_on_dark,
                    diameter_bins = diameter_bins,
                    optimize = False
                )
                
                #Paste the data into the output dataframe       
                img_df = pd.concat((img_df, temp_df))
                
                #save images?
                if save_images is True:
                    #Convert boolean array to 8-bit, 1-band image. Requires PIL.
                    im = Image.fromarray((255*~objects).astype('uint8')) #black roots, white background
                    im.save(root_directory + os.sep + "Pyroots Analyzed" + os.sep + file_in)
            
    return(img_df)
