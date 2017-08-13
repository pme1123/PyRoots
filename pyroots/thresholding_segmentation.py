"""
Created 08/12/2016

@author: pme

Contents:
- pyroots_analysis
- image_loop

TODO: Test multi_image_loop
"""

from pyroots import *
from skimage import io, color, filters, exposure, img_as_ubyte
from warnings import warn

def thresholding_segmentation(image,
                              threshold_args,
                              image_name='Default Image',
                              colors='dark',
                              contrast_kernel_size='skip',
                              mask_args='skip',
                              noise_removal_args='skip',
                              morphology_filter_args='skip',
                              fill_gaps_args='skip',
                              lw_filter_args='skip',
                              diam_filter_args='skip',
                              diameter_bins=None,
                              verbose=False):
    """
    Full analysis of an image for length of objects based on thresholding.
    Performs the following steps:
    1. Colorspace conversion and selecting analysis bands
    2. Contrast enhancement
    3. Adaptive thresholding bands to binary images
    4. Combining multiplicatively (i.e. kept if `True` in all, if multiple bands)
    5. Filter objects by size, length:width ratio, and diameter (all optional)
    6. Smoothing (optional)
    7. Measuring medial axis length and diameter along the length
    8. Summarizing by diameter class or the entire image

    Methods that are optional are set as 'skip' for default. Most arguments require
    dictionaries of arguments for the subfunctions. The easiest way to generate these
    dictionaries is to use the thresholding-based segmentation notebook. See the pyroots
    functions for more information.

    Parameters
    ----------
    image : array
    	An RGB or black and white image for analysis

    threshold_args : list of dicts
        Dictionaries contain options for adaptive thresholding. See skimage.filters.threshold_adaptive().
        At minimum, requires 'block_size', for example, threshold_args = [{'block_size':101}].

    image_name : str
    	What do you want to call your image?

    colors : dict or string
        See `pyroots.band_selector`
        For color analysis:
            Currently only supports one colorspace, but you can choose multiple bands.
        	A dictionary containing:
            - colorspace: string.
                Colorspace in which to run the analysis (ex. RGB, LAB, HSV). See
                scikit-image documentation for all options.
            - band: list of integers
                Specifying which bands of `colorspace` on which to run the analysis
                (ex. 2 in RGB gives Blue, 0 gives Red).
            - dark_on_light: list of boolean
                Are the objects dark objects on a light background? Length must match
                length of `colors['band']`.
        For black and white analysis:
            A string of either:
                `'dark'` for dark roots on a light background
                `'light'` for light roots on a dark background

    contrast_kernel_size : int or None
        Dimension of kernel for adaptively enhancing contrast. Calls
        `skimage.exposure.equalize_adapthist()`. If `None`, will use a default of
        1/8 height by 1/8 width.

    mask_args : dict
    	Used for masking the image with an ellipse. Useful for photomicroscopy. See `pr.ellipse_mask`.

    noise_removal_args : dict
    	Smooths and despeckles the image, and also separates loosely connected objects for easier filtering.
        Contains arguments for `pyroots.noise_removal()`.

    morphology_filter_args : dict
        Filters objects by shape, size, and solidity. See `pyroots.morphology_filter()`.

    fill_gaps_args : dict
        Removes small holes and gaps between objects, now that most noise is removed. See
        `pyroots.fill_gaps()`.

    lw_filter_args : dict
        Removes objects based on medial axis length:mean width ratios. See `pyroots.length_width_filter()`.

    diam_filter_args : dict
        Removes entire objects or parts of objects based on diameters. See `pyroots.diameter_filter()`.

    diameter_bins : list of float
    	Bin cutoffs for summarizing object length by diameter class.
    	Defaults to `None`, which returns total length and average diameter for
    	all objects in the image.

    verbose : bool
        Give feedback showing the step working on?

    Returns
    -------
    A dictionary containing:
        1. 'geometry' : a `pandas` dataframe describing either:
            - image name, total length, mean diameter, and the number of objects (if `diameter_bins` is `None`)
            - image name, length by diameter class, and diameter class (otherwise)
        2. 'objects'  : a binary image of kept objects
        3. 'length'   : a 2D image array of object medial axes with values indicating the length at that axis
        4. 'diameter' : a 2D image array of object medial axes with values indicating either:
            - the diameter at that pixel (if `diameter_bins` is `None`)
            - the diameter bin to which a pixel belongs (otherwise)
    3) skeleton pixel lengths; 4) skeleton pixel diameters.
    
    Notes
    -----
    Most functions within this method are attempted. If they receive an unuseable argument, 
    e.g. the dictionary contains a formatting error or a bad keyword, then the method will be 
    skipped with a warning. If you see such a warning ("Skipping (function)..."), check the 
    formatting of your argument and re-create a parameters file with a jupyter notebook.
    
    See Also
    --------
    For example parameter dictionaries, see example_thresholding_analysis_parameters.py.

    """


    # Begin
    ## Convert Colorspace, enhance contrast
    # Pull band from colorspace
    working_image = band_selector(image, colors)
    nbands = len(working_image)
    if verbose is True:
        print("Color bands selected")
    

    ## Count nubmer of dictionaries in threshold_args. Should equal number of bands. Make sure is list.
    try:
        len(threshold_args[0])
    except:
        threshold_args = [threshold_args]
        if nbands != len(threshold_args):
            raise ValueError("Number of dictionaries in `threshold_args` doesn't\
                             equal the number of bands in `colors['band']`!")
        pass        

    try:
        for i in range(nbands):
            temp = exposure.equalize_adapthist(working_image[i],
                                               kernel_size = contrast_kernel_size)
            working_image[i] = img_as_ubyte(temp)
        if verbose is True:
            print("Contrast enhanced")
    except:
        if contrast_kernel_size is not 'skip':
            warn("Skipping contrast enhancement")
        pass

    ## threshold
    for i in range(nbands):
        working_image[i] = working_image[i] > filters.threshold_local(working_image[i],
                                                                      **threshold_args[i])
    for i in range(nbands):
        if len(colors) == 3:
            if colors['dark_on_light'][i] is True:
                working_image[i] = ~working_image[i]
        else:
            if colors == 'dark':
                working_image[i] = ~working_image[i]
                
    ## Combine bands. As written, keeps all 'TRUE'
    combined = working_image[0].copy()
    for i in range(1, nbands):
        combined = combined * working_image[i]

    working_image = combined.copy()
    if verbose is True:
        print("Thresholding complete")

    ## Mask, filtering, smoothing
    try:
        working_image = working_image * draw_mask(working_image, **mask_args)
        if verbose is True:
            print("Image masked")
    except:
        if mask_args is not 'skip':
            warn("Skipping mask")
    pass

    try:
        working_image = noise_removal(working_image, **noise_removal_args)
        if verbose is True:
            print("Smoothing and noise removal complete")
    except:
        if noise_removal_args is not 'skip':
            warn("Skipping noise removal")
        pass

    try:
        working_image = morphology_filter(working_image, **morphology_filter_args)
        if verbose is True:
            print("Morphology filtering complete")
    except:
        if morphology_filter_args is not 'skip':
            warn("Skipping morphology filter")
        pass

    try:
        working_image = fill_gaps(working_image, **fill_gaps_args)
        if verbose is True:
            print("Smoothing and gap filling complete")
    except:
        if fill_gaps_args is not 'skip':
            warn("Skipping gap filling and smoothing")
        pass

    ## skeleton, length-width, diameter filters
    skel_dict = skeleton_with_distance(working_image)
    if verbose is True:
        print("Skeletonization complete")

    try:
        lw_dict = length_width_filter(skel_dict, **lw_filter_args)
        if verbose is True:
            print("Length:width filtering complete")
    except:
        lw_dict = skel_dict.copy()
        if lw_filter_args is not 'skip':
            warn("Skipping length-width filter")
        pass

    try:
        diam_dict = diameter_filter(lw_dict, **diam_filter_args).copy()
        if verbose is True:
            print("Diameter filter complete")
    except:
        if diam_filter_args is not 'skip':
            warn("Skipping diameter filter")
        pass

    ## Summarize
    if diameter_bins is None or diameter_bins is 'skip':
        summary_df = summarize_geometry(skel_dict['geometry'], image_name)

    else:
        diam_out, summary_df = bin_by_diameter(skel_dict['length'],
                                               skel_dict['diameter'],
                                               diameter_bins,
                                               image_name)
        skel_dict['diameter'] = diam_out

    out = {'geometry' : summary_df,
           'objects'  : skel_dict['objects'],
           'length'   : skel_dict['length'],
           'diameter' : skel_dict['diameter']}

    if verbose is True:
        print("Done")

    return(out)
