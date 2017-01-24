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
from pyroots.image_manipulation import img_split
from pyroots.geometry_filters import morphology_filter, hollow_filter
from pyroots.noise_filters import color_filter
from pyroots.summarize import bin_by_diameter, summarize_geometry
from pyroots.skeletonization import skeleton_with_distance
from skimage import io, color, filters, morphology, img_as_ubyte
from multiprocessing import Pool  
from multiprocessing.dummy import Pool as ThreadPool

def frangi_segmentation(image, colors, frangi_args, threshold_args,
                        color_args_1=None, color_args_2=None, 
                        morphology_args=None, hollow_args=None, hole_filling=None, 
                        diameter_bins=None, image_name="image"):
    """
    Possible approach to object detection using frangi filters. Selects colorbands for
    analysis, runs frangi filter, thresholds to identify candidate objects, then removes
    spurrious objects by characteristics. See frangi_approach.ipynb. 
    
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
    morphology_args : dict
        Parameters to pass to `pyroots.morphology_filter`
    hollow_args : dict
        Parameters to pass to `pyroots.hollow_filter`
    hole_filling : dict
        'selem' : ndarray to pass to `morphology.binary_closing`
        'min_size' : int to pass to `morphology.remove_small_holes`
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
    
    # Frangi vessel enhancement
    working_image = filters.frangi(working_image, **frangi_args)
    working_image = 1-(working_image/working_image.max())  # rescale and invert --> dark objects for thresholding

    # Threshold to ID candidate objects
    working_image = filters.threshold_adaptive(working_image, **threshold_args)
    working_image = ~working_image  # new line for clarity; --> objects = True
    
    # Filter candidate objects by color
    try:
        working_image = color_filter(image, working_image, **color_args_1)  #colorspace, target_band, low, high, percent)
        
        try:
            working_image = color_filter(image, working_image, **color_args_2)  # nesting equates to an "and" statement.
        except:
            pass
    except:
        pass
    
    # Filter candidate objects by morphology
    try:
        working_image = morphology_filter(working_image, **morphology_args)  #### ADD PR
    except:
        pass
    
    # Filter candidate objects by hollowness
    try:  
        working_image = hollow_filter(working_image, **hollow_args)          #### ADD PR
    except:
        pass
    
    # Close small gaps and holes in accepted objects
    try:
        working_image = morphology.binary_closing(working_image, selem=hole_filling['selem'])
        working_image = morphology.remove_small_holes(working_image, min_size=hole_filling['min_size'])
    except:
        pass
    
    # Skeletonize
    skel = skeleton_with_distance(working_image)
    
    # Summarize
    if diameter_bins is None:
        summary_df = summarize_geometry(skel['geometry'], image_name)

    else:
        diam_out, summary_df = bin_by_diameter(skel['length'],
                                                  skel['diameter'],
                                                  diameter_bins,
                                                  image_name)
        skel['diameter'] = diam_out
    
    out = {'geometry' : summary_df,
           'objects'  : skel['objects'],
           'length'   : skel['length'],
           'diameter' : skel['diameter']}
    
    return(out)
    
    
##################################################################################################################################################################################
####################                                                                                                                               ###############################
####################                                                 IMAGE LOOP - IN SERIES                                                        ###############################
####################                                                                                                                               ###############################
##################################################################################################################################################################################

    
def frangi_image_loop(base_directory, image_extension, params=None, out_dir="Pyroots Analyzed", mask=None, save_images=False, threads=1):
    """
    Reference function to loop through images in a directory. As it is written, it returns
    a dataframe from `pyroots.frangi_segmentation` and also writes images showing the objects analyzed.
    Note that all parameters in `pyroots.frangi_segmentation` must be defined in the params
    file, even as `None` for this to work.
    
    Use the parallel version for lots of images, if you have the capability (i.e. supercomputer or 4+ cores)
    
    Parameters
    ----------
    base_directory : str
        Directory to the images or subdirectories containing the images. 
    image_extension : str
        Extension of images to analyze
    params : str
        Path + filename for parameters file for `pyroots.frangi_segmentation`. If None (default), define the parameters
        in the workspace using appropriately named dictionaries. 
    out_dir : str
        Name of directory to write output images
    mask : ndarray
        TODO: For limiting the analysis
    save_images : bool
        Do you want to save images of the objects?
    threads : int
        For multiprocessing
    """
    
    try:
        exec(open(params).read())
    except:
        pass
    
    #Make an output directory for the analyzed images and data.
    if save_images is True:
        if not os.path.exists(base_directory + os.sep + out_dir):
            os.mkdir(base_directory + os.sep + out_dir)
    
    #Make a placeholder DataFrame for the output data. Requires pandas.
    if diameter_bins is None:
        img_df = pd.DataFrame(columns=("ImageName", "Length", "NObjects", "MeanDiam"))    
    else:
        img_df = pd.DataFrame(columns=("ImageName", "DiameterClass", "Length"))
    
    out = []
    #Begin looping
    for subdir, dirs, files in os.walk(base_directory):        
        # What we'll do:        
        def _core_fn(file_in):
            if file_in.endswith(image_extension) and not subdir.endswith(out_dir):
                time_in = current_time()
                                
                path_in = subdir + os.sep + file_in  # what's the image called and where is it?
                file_name = subdir[len(base_directory):] + file_in

                #Create filein name
                filein = "DIR" + subdir[len(base_directory):] + os.sep + file_in
                #Import image
                img = io.imread(path_in)[:,:,0:3]  # load image

                #Run through the analysis. Requires pyroots.pyroots_analysis reference function.    ###############################################################
                objects_dict = frangi_segmentation(img, colors, frangi_args,                      ######   THIS IS WHERE YOU INSERT YOUR CUSTOM FUNCTION   ######
                                                      threshold_args, color_args_1,                    ###############################################################
                                                      color_args_2, morphology_args, 
                                                      hollow_args, hole_filling, 
                                                      diameter_bins, image_name=file_name)
                #save images?
                if save_images is True:
                    #Convert boolean array to 8-bit, 1-band image. Requires PIL
    #                im = Image.fromarray((255 * objects_dict['objects']).astype('uint8')) #white roots, black background    #### MAY NEED TO UPDATE THIS. 
                    path_out = base_directory + os.sep + out_dir + os.sep + subdir[len(base_directory):] + file_in 
                    io.imsave(path_out, 255*objects_dict['objects'].astype('uint8'))
                
                print("Done: " + subdir[len(base_directory):] + file_in)
                df_out = objects_dict['geometry']
                return(df_out)

        # Init threads within each subdir (at file level)
        thread_pool = ThreadPool(threads)
        # Work on _core_fn
        out += thread_pool.map(_core_fn, files)
        # finish
        thread_pool.close()
        thread_pool.join()
    
    out = pd.concat([i for i in out])
    
    #Done        
    return(out)
