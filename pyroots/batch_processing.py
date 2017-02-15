"""
Batch processing functions:
- preprocessing_filter_loop
- preprocessing_manipulation_loop

"""


import os
import numpy as np
from pyroots import preprocessing_filters
from skimage import io
from multiprocessing import Pool  
from multiprocessing.dummy import Pool as ThreadPool

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
    
    for path, folder, file in os.walk(dir_in):  
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
                                                     missing_band_params,
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
            thread_pool.map(_core_fn, file)
            # finish
            thread_pool.close()
            thread_pool.join()
        
    return("Done")
