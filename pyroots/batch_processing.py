"""
Batch processing functions:
- preprocessing_filter_loop
- preprocessing_manipulation_loop

"""


import os
import numpy as np
from numpy import array, uint8
from pyroots import preprocessing_filters, preprocessing_actions, frangi_segmentation
import pandas as pd
from pyroots.image_manipulation import img_split
from pyroots.geometry_filters import morphology_filter, hollow_filter, diameter_filter
from pyroots.noise_filters import color_filter
from pyroots.summarize import bin_by_diameter, summarize_geometry
from pyroots.skeletonization import skeleton_with_distance
from skimage import io, color, filters, morphology, img_as_ubyte, img_as_float
from skimage import io
from multiprocessing import Pool  
from multiprocessing.dummy import Pool as ThreadPool
from warnings import warn   
import cv2


#################################################################################################
#################################################################################################
#########                                                                           #############
#########                      Preprocessing Image Filtering Loop                   #############
#########                                                                           #############
#################################################################################################
#################################################################################################

def preprocessing_filter_loop(dir_in, 
                              extension_in, 
                              dir_out,
                              extension_out=".png",
                              params=None,
                              threads=1):
    """
    Combines preprocessing filters (blur, color, contrast) into a loop. Convenient to run as a vehicle to transfer images
    from a portable drive to a permanent area. 
    
    Parameters
    ----------
    base_directory : str
        Directory to the images or subdirectories containing the images. 
    extension_in : str
        Extension of images to analyze
    out_dir : str
        Name of directory to write output images. `None` to skip.
    extension_out : str
        Extension to save images.
    params : str
        Path + filename for parameters file for `pyroots.preprocessing_filters`. If `None` (default), only
        loads and (possibly) resaves images. If not `None`, will only save images that pass test. See notes for format.
    threads : int
        For multiprocessing
    
    Returns
    -------
    Image files.
        
    Notes
    -----
    The `params` file should have the following objects: `blur_params`, `missing_band_params`, and `low_contrast_params`.
    Except for `blur_band`, all are dictionaries with items named as arguments in respective functions. If not present, 
    defaults to `None`. Will raise a `UserWarning` if the format and names are not correct (but not if `None`). 
    
    """
    
    
    try:
        exec(open(params).read(), globals())
    except:
        if params is not None:
            raise ValueError("Couldn't load params file. Try checking it for words like 'array' or\n'uint8' that need to be loaded with numpy and load these\n functions using `extra_imports`.... Or edit source.")
    
    for path, folder, filename in os.walk(dir_in):  
        if dir_out not in path:   # Don't run in the output directory.
            
            # Make directory for saving objects
            subpath = path[len(dir_in)+1:]
            
            if not os.path.exists(os.path.join(dir_out, subpath)):
                os.mkdir(os.path.join(dir_out, subpath))
                os.mkdir(os.path.join(dir_out, "DID NOT PASS", subpath))

            def _core_fn(filename):
                if filename.endswith(extension_in):
                    path_in = os.path.join(path, filename)  # what's the image called and where is it?
                    
                    # possible locations to write the output file
                    filename_out = os.path.splitext(filename)[0] + extension_out
                    path_out_PASS = os.path.join(dir_out, subpath, filename_out)
                    path_out_FAIL = os.path.join(dir_out, "DID NOT PASS", subpath, filename_out)
                    
                    # skip if already analyzed
                    if os.path.exists(path_out_PASS) or os.path.exists(path_out_FAIL):
                        print("SKIPPING: {}".format(os.path.join(subpath, filename)))
                    
                    else:  # analyze
                        try:
                            img = io.imread(path_in)  # load image
                        except:
                            print("Couldn't load: {}. Continuing...".format(path_in))
                            filename_out = "MISLOAD" + filename_out
                            path_out_FAIL = os.path.join(dir_out, "DID NOT PASS", subpath, filename_out)
                            img = np.ones((30, 30, 3))  # make an image that cannot pass
                            pass
                        
                        test = preprocessing_filters(img,
                                                     blur_params,
                                                     temperature_params,
                                                     low_contrast_params,
                                                     center)
                        
                        # where to write the output file?
                        if test == True:
                            if dir_out is not None:
                                io.imsave(path_out_PASS, img)
                            print("PASSED: {}".format(os.path.join(subpath, filename)))

                        else:
                            if dir_out is not None:
                                io.imsave(path_out_FAIL, img)
                            print("DID NOT PASS: {}".format(os.path.join(subpath, filename)))
                
            thread_pool = ThreadPool(threads)
            # Work on _core_fn
            thread_pool.map(_core_fn, filename)
            # finish
            thread_pool.close()
            thread_pool.join()
        
    return("Done")
    
#################################################################################################
#################################################################################################
#########                                                                           #############
#########                      Preprocessing Actions Loop                           #############
#########                                                                           #############
#################################################################################################
#################################################################################################

def preprocessing_actions_loop(dir_in, 
                               extension_in, 
                               dir_out,
                               extension_out=".png",
                               params=None,
                               threads=1):
    """
    Combines preprocessing filters (blur, color, contrast) into a loop. Convenient to run as a vehicle to transfer images
    from a portable drive to a permanent area. 
    
    Parameters
    ----------
    base_directory : str
        Directory to the images or subdirectories containing the images. 
    extension_in : str
        Extension of images to analyze
    out_dir : str
        Name of directory to write output images. `None` to skip.
    extension_out : str
        Extension to save images.
    params : str
        Path + filename for parameters file for `pyroots.preprocessing_filters`. If `None` (default), only
        loads and (possibly) resaves images. If not `None`, will only save images that pass test. See notes for format.
    threads : int
        For multiprocessing
    
    Returns
    -------
    Image files.
        
    Notes
    -----
    The `params` file should have the following objects: `make_correction_params`, 
    `brightfield_correction_params`, 'registration_params', and `low_contrast_params`.
    All are dictionaries with items named as arguments in respective functions. If not present, 
    will default to `None`. Will raise a `UserWarning` if the format and names are not correct (but not if `None`). 
    
    """
    
    
    try:
        exec(open(params).read(), globals())
    except:
        if params is not None:
            raise ValueError("Couldn't load params file. Try checking it for words like 'array' or\n'uint8' that need to be loaded with numpy and load these\n functions using `extra_imports`.... Or edit source.")
            return  # can't do anything without hte parameters!
    
    # make sure all dictionaries have something assigned to them, including None
    dicts = ['make_brightfield_params', 
             'brightfield_correction_params',
             'smoothing_params',
             'registration_params']
    print("The parameters you've loaded are:\n")
    
    for i in dicts:
        try:
            print(i + " = " + str(globals()[i]))
        except:
            print(i + " = " + str(None))
            globals()[i] = None
    
    # initiate loop
    for path, folder, filename in os.walk(dir_in):  
        if dir_out not in path:   # Don't run in the output directory.
            # ID the current folder 
            subpath = path[len(dir_in)+1:]

            # Make directories for saving images
            if dir_out is not None:
                if not os.path.exists(os.path.join(dir_out, subpath)):
                    os.mkdir(os.path.join(dir_out, subpath))
                    os.mkdir(os.path.join(dir_out, "FAILED PROCESSES", subpath))
                    
            # make a brightfield correction image for the directory
            def _make_brightfield_image(directory, brightfield_name, brightfield_sigma):
                correction = io.imread(os.path.join(directory, brightfield_name))
                correction = cv2.GaussianBlur(correction, (0, 0), brightfield_sigma)
                return(correction)
            try:
                correction = _make_brightfield_image(path, **make_brightfield_params)
            except:
                if make_brightfield_params is not None:
                    print("\nCould not make correction image. Does\n{}\nexist?\nContinuing to next folder...\n".format(\
                        os.path.join(subpath, make_brightfield_params['brightfield_name'])))
                    continue  # can't do this folder, so move on to the next
                else:
                    correction = None
                    pass
            
            
            def _core_fn(filename):
                if filename.endswith(extension_in):
                    path_in = os.path.join(path, filename)  # what's the image called and where is it?
                    
                    # possible locations to write the output file
                    filename_out = os.path.splitext(filename)[0] + extension_out
                    path_out_PASS = os.path.join(dir_out, subpath, filename_out)
                    path_out_FAIL = os.path.join(dir_out, "FAILED PROCESSES", subpath, filename_out)
                    
                    # skip if already analyzed
                    if os.path.exists(path_out_PASS) or os.path.exists(path_out_FAIL):
                        print("Already Analyzed: {}".format(os.path.join(subpath, filename)))
                    
                    else:  # analyze
                        try:
                            img = io.imread(path_in)  # load image
                        except:
                            print("Couldn't load: {}. Continuing...".format(path_in))
                            return
                        
                        img_out, warnings = preprocessing_actions(img, 
                                                                  correction, 
                                                                  brightfield_correction_params, 
                                                                  registration_params, 
                                                                  smoothing_params, 
                                                                  count_warnings=True)
                        
                        # where to write the output file?
                        if warnings == 0:  # save the manipulated image
                            if dir_out is not None:
                                io.imsave(path_out_PASS, img_out)
                            print("PASSED: {}".format(os.path.join(subpath, filename)))

                        else:
                            if dir_out is not None:  # save a copy of the preprocessed image
                                io.imsave(path_out_FAIL, img)
                            print("Something Failed: {}".format(os.path.join(subpath, filename)))
                
            thread_pool = ThreadPool(threads)
            # Work on _core_fn
            thread_pool.map(_core_fn, filename)
            # finish
            thread_pool.close()
            thread_pool.join()
        
    return("Done")


##################################################################################################################################################################################
####################                                                                                                                               ###############################
####################                                               FRANGI-BASED SEGMENTATION                                                       ###############################
####################                                                                                                                               ###############################
##################################################################################################################################################################################


def frangi_image_loop(dir_in, 
                      extension_in, 
                      dir_out=None, 
                      table_out=None,
                      table_overwrite=False,
                      params=None,
                      mask=None,
                      save_images=False,
                      threads=1):
    """
    Reference function to loop through images in a directory. As it is written, it returns
    a dataframe from `pyroots.frangi_segmentation` and also writes images showing the objects analyzed.
    Note that all parameters in `pyroots.frangi_segmentation` must be defined in the params
    file, even as `None` for this to work.
    
    Parameters
    ----------
    dir_in : str
        Directory to the images or subdirectories containing the images. 
    extension_in : str
        Extension of images to analyze
    dir_out : str or `None`
        Full path to write images to. If `None` and `save_images=True`, defaults to "Pyroots Analyzed" in `dir_in`.
    table_out : str or `None`
        Full path to write table to. If `None`, defaults to "Pyroots Results.txt" in `dir_in`. 
    table_overwrite : bool
        If `table_out` exists, do you want to overwrite it?
    params : str
        Path + filename for parameters file for `pyroots.frangi_segmentation`. If None (default), will 
        print a list of subpaths + images that would be processed, with a warning. 
    mask : ndarray
        Binary array of the same dimensions as each image, with 1 being the part of the image to analyze. 
    save_images : bool
        Do you want to save images of the objects?
    threads : int
        For multiprocessing
    extra_imports : list
        If raises error importing params, then write a list of lists as
            [[lib1, fun1, fun2, ...], 
             [lib2, fun1, fun2, ...], 
             [...]]. 
    """    
    
    if params is not None:
        try:
            exec(open(params).read(), globals())
        except:
            if os.path.exists(params):
                raise ValueError("Couldn't load params file. Try checking it for words like 'array' or\
                \n'uint8' that need to be loaded with numpy and load these functions\
                \n at the top of your script.... Or edit source.")
            else:
                raise ValueError("Couldn't find params file at {}".format(params))
    else:
        print("No parameters defined. Printing paths to images.")
    
    # make directories out
    if dir_out is None:
        dir_out = os.path.join(dir_in, "Pyroots Analyzed")
    if table_out is None:
        table_out = os.path.join(dir_in, "Pyroots Results.txt")
    
    #Make an output directory for the analyzed images and data.
    if save_images is True:
        if params is not None:
            print("Saving images to {}".format(dir_out))
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)
    
    #Test for table overwrite
    if os.path.exists(table_out):
        if table_overwrite is False:
            raise ValueError("Data output table already exists! Aborting...")
        else:
            print("Overwriting old data table: {}".format(table_out))
    else:
        print("Saving data table to: {}".format(table_out))
    
    #Begin looping
    out = []
    for path, folder, filename in os.walk(dir_in):  
        if dir_out not in path:   # Don't run in the output directory.
            
            # Make directory for saving objects
            subpath = path[len(dir_in)+1:]
            if not os.path.exists(os.path.join(dir_out, subpath)):
                os.mkdir(os.path.join(dir_out, subpath))
                os.mkdir(os.path.join(dir_out, "FAILED", subpath))
            
            # What we'll do:        
            def _core_fn(filename):
                if filename.endswith(extension_in):

                    path_in = os.path.join(path, filename)
                    # Where to write the output image?
                    filename_out = os.path.splitext(filename)[0] + ".png"
                    path_out = os.path.join(dir_out, subpath, filename_out)
                    
                    if os.path.exists(path_out): #skip
                        print("ALREADY ANALYZED: {}. Skipping...".format(os.path.join("..", subpath, filename)))
                    
                    else: #(try to) do it
                        try:
                            img = io.imread(path_in)  # load image
                            if mask is not None:
                                img = img * mask
                            
                            objects_dict = frangi_segmentation(img, colors,              #### Insert your custom function here ####
                                                               frangi_args, 
                                                               threshold_args, 
                                                               color_args_1,
                                                               color_args_2, 
                                                               color_args_3,
                                                               morphology_args, 
                                                               hollow_args,
                                                               fill_gaps_args,
                                                               diameter_args,
                                                               diameter_bins, 
                                                               image_name=os.path.join(subpath,
                                                                                       filename))
                            #save images?
                            if save_images is True:
                                io.imsave(path_out, 255*objects_dict['objects'].astype('uint8'))
                            print("Done: {}".format(os.path.join(subpath, filename)))
                            df_out = objects_dict['geometry']
                        
                        except:
                            if params is None:
                                print(os.path.join(subpath, filename))
                                df_out = pd.DataFrame(columns=("ImageName", "DiameterClass", "Length"))
                            elif diameter_bins is None:
                                print("Couldn't Process: {}.\n     ...Continuing...".format(path_in))
                                df_out = pd.DataFrame(columns=("ImageName", "Length", "NObjects", "MeanDiam"))    
                            else:
                                print("Couldn't Process: {}.\n     ...Continuing...".format(path_in))
                                df_out = pd.DataFrame(columns=("ImageName", "DiameterClass", "Length"))
                        return(df_out)

            # Init threads within each path
            if threads is None:
                out += map(_core_fn, filename)
            else:
                thread_pool = ThreadPool(threads)
                # Work on _core_fn
                out += thread_pool.map(_core_fn, filename)
                # finish
                thread_pool.close()
                thread_pool.join()
            
    
    out = pd.concat([i for i in out])
    
    try:
        out.to_csv(table_out, sep="\t", index=False) 
    except:
        print(
            "****************************\n********  WARNING  *********\n****************************\
            \n\nCould not export results to\n{}!!\n\n****************************\n********  WARNING  *\
            ********\n****************************".format(table_out))
        pass
    #Done        
    return(out)