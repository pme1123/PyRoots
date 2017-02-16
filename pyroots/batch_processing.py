"""
Batch processing functions:
- preprocessing_filter_loop
- preprocessing_manipulation_loop

"""


import os
import numpy as np
from pyroots import preprocessing_filters, preprocessing_actions
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
                print("\nCould not make correction image. Does\n{}\nexist?\nContinuing to next folder...\n".format(\
                     os.path.join(subpath, make_correction_params['brightfield_name'])))
                continue  # can't do this folder, so move on to the next
            
            
            
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
