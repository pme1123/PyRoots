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
from pyroots import *
from skimage import io, color, filters, exposure, img_as_ubyte
from warnings import warn

def thresholding_segmentation(image,
                              threshold_args,
                              image_name='Default Image',
                              colors='skip',
                              contrast_kernel_size='skip',
                              mask_args='skip',
                              noise_removal_args='skip',
                              morphology_filter_args='skip',
                              fill_gaps_args='skip',
                              lw_filter_args='skip',
                              diam_filter_args='skip',
                              diameter_bins=None,
                              verbose=False):
    #TODO: Test
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

    colors : dict
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
        `skip` for a black and white image.

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

    See Also
    --------
    For example parameter dictionaries, see example_thresholding_analysis_parameters.py.

    """

    # Housekeeping
    ## count number of bands and make sure the items 'band' and 'dark_on_light' are lists
    try:
        nbands = len(colors['band'])
    except:
        colors['band'] = [colors['band']]
        nbands = len(colors['band'])
        pass

    try:
        len(colors['dark_on_light'])
    except:
        colors['dark_on_light'] = [colors['dark_on_light']]
        if nbands != len(colors['dark_on_light']):
            raise ValueError("Number of items in `colors['dark_on_light']` doesn't\
                             equal the number of bands in `colors['band']`!")
        pass

    ## Count nubmer of dictionaries in threshold_args. Should equal number of bands. Make sure is list.
    try:
        len(threshold_args[0])
    except:
        threshold_args = [threshold_args]
        if nbands != len(threshold_args):
            raise ValueError("Number of dictionaries in `threshold_args` doesn't\
                             equal the number of bands in `colors['band']`!")
        pass

    # Begin
    ## Convert Colorspace, enhance contrast
    try:  # detects if color=None
        if colors['colorspace'].lower() != 'rgb':
            working_image = getattr(color, "rgb2" + colors['colorspace'])(image)
        else:
            working_image = image.copy()
        working_image = [img_split(working_image)[i] for i in colors['band']] # pull bands
    except:  # for black and white
        if colors is not 'skip':
            raise ValueError("Your colors arguments are invalid. For black and white, set colors=None.")
        working_image = image.copy()
        pass
    if verbose is True:
        print("Analysis bands selected")

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
        if colors['dark_on_light'][i] is True:
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
    if diameter_bins is None:
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

def thresholding_image_loop(root_directory, image_extension, path_to_params, mask=None, save_images=False):
    """
    Reference function to loop through images in a directory. As it is written, it returns
    a dataframe from "pyroots_analysis" and also writes images showing the objects analyzed.
    """
    exec(open(path_to_params).read())
    #Make an output directory for the analyzed images and data. Requires os.
    if save_images is True:
        if not os.path.exists(root_directory + os.sep + "Pyroots Analyzed"):
            os.mkdir(root_directory + os.sep + "Pyroots Analyzed")

    #Make a placeholder DataFrame for the output data. Requires pandas.
    if diameter_bins is None:
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
                    image_name = file_in,
                    colorspace = colorspace,
                    analysis_bands = analysis_bands,
                    threshold_params = threshold_params,
                    mask = mask,
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
