"""
Functions to sort and prepare raw input images for analysis with pyroots functions. 

Contents:
- detect_motion_blur
- detect_missing_bands
- correct_brightfield
- register_bands
- preprocessing_filters
"""
import numpy as np
from skimage import filters, img_as_ubyte, exposure
from pyroots import img_split, _center_image
import cv2
from warnings import warn


###############################################################################################
#########                                                                        ##############
#########                               Blur Detector                            ##############
#########                                                                        ##############
###############################################################################################

def detect_motion_blur(image, ratio=2, band=None):
    """
    Detect if an image is blurry due to movement, and therefore unreliable
    to analyze. Uses the variance of the prewitt edge detectors in the horizontal and vertical directions.
    
    Parameters
    ----------
    image : ndarray
        rgb or grayscale image. 
    ratio : float
        Maximum ratio of edge strength variance in x- to y-directions. `threshold=2` means if
        `x_edge.var()` / `y_edge.var() is > 2 OR < 0.5, the image is flagged as blurry do to
        movement.
    band : int
        Band of image to use for analysis. Unspecified defaults to 0 (R in RGB). 
        
    Returns
    -------
    bool - does the image meet requirements?
    
    """
    if band is None:
        try:
            analyze = image[:, :, 0].copy()
        except:
            analyze = image.copy()
            pass
    else:
        analyze = image[:, :, band]

    horiz = img_as_ubyte(filters.prewitt_v(analyze)).var()
    vert = img_as_ubyte(filters.prewitt_h(analyze)).var()
    
    test = horiz / vert
    
    if test < ratio and test > 1/ratio:
        out = True
    else:
        out = False
    
    return(out)


###############################################################################################
#########                                                                        ##############
#########                           Missing Band Detector                        ##############
#########                                                                        ##############
###############################################################################################

def detect_missing_bands(image, percentile=90, min_value=0.0):
    """
    Detecs whether an image is missing a band. This happens with some camera processing software.
    
    Parameters
    ----------
    image : ndarray
        RGB image
    percentile : float
        percentile of values in each band to test against the threshold. Default 90.
    min_value : float
        floor of values at each band to pass. Default 0.0
    
    Returns
    -------
    bool - Does the image pass criteria?
    """
    tests = img_split(image)
    tests = [np.percentile(i, percentile) > min_value for i in tests]
    
    return(sum(tests) == 3)



###############################################################################################
#########                                                                        ##############
#########                          Brightfield Correction                        ##############
#########                                                                        ##############
###############################################################################################
def correct_brightfield(image, brightfield, correction_factor=1):
    """
    Adjusts exposure of an image based on a 'brightfield' blank. This corrects exposure vignetting
    (dark edges, bright centers, for example) of images. Often favorably enhances color. 
    
    Parameters
    ----------
    image : ndarray
        image of same shape as the brightfield
    brightfield : ndarray
        image of 'blank' background, probably with gaussian blur added.
    correction_factor : float, int
        scale brightfield values to reduce/increase saturation. See notes.
    
    Returns
    -------
    an ndarray of shape image.
    
    Notes
    -----
    This function divides `image` by `brightfield`. If `brightfield` is darker than `image` at
    some pixels, then values are outside of the normal range of images. If this happens excessively, 
    the corrected image will look oversaturated. In this case, increase `correction_factor` slightly 
    to scale up `brightfield`. Likewise, if the corrected image is undersaturrated, decrease 
    `correction_factor` slightly. This function sets a ceiling of output values at 255. 

    """
    out = image / (brightfield * correction_factor)
    out[out>1] = 1

    out = img_as_ubyte(out)
    return(out)
    
###############################################################################################
#########                                                                        ##############
#########                             Band Registration                          ##############
#########                                                                        ##############
###############################################################################################
def register_bands(image, template_band=1, ECC_criterion=True):
    """
    Fix chromatic abberation in images by calculating and applying an affine
    transformation. Chromatic abberation is a result of uneven refraction of light
    of different wavelengths. It shows up as systematic blue and red edges in 
    high-contrast areas.
    
    This should be done before other processing to minimize artifacts.
    
    Parameters
    ----------
    image : ndarray
        3- or 4-channel image, probably RGB.
    template : int
        Band to which to align the other bands. Usually G in RGB. 
    ECC_criterion : bool
        Use ECC criterion to find optimal warp? Improves results, but increases
        processing time 5x. 
    
    Returns
    -------
    An ndarray of `image.size`
    
    Notes
    -----
    Uses `skimage.filters.scharr` to find edges in each band, then finds and
    applies an affine transformation to register the images using 
    `cv2.estimateRigidTransform` and `cv2.warpAffine`. If `ECC_criterion=True`,
    the matrix from `estimateRigidTransform` is updated using `cv2.findTransformECC`. 
    """
    
    #find dimensions
    height, width, depth = image.shape
    
    #define bands to analyze
    analyze = []
    for i in range(depth):
        if i != template_band:
            analyze.append(i)
    
    # Extract bands, find edges
    bands = img_split(image)
    edges = [img_as_ubyte(filters.scharr(i)) for i in bands]

    #make output image
    out = np.zeros((height, width, depth), dtype=np.uint8)
    out[:, :, template_band] = bands[template_band]
    
    for i in analyze:
        # Estimate transformation
        warp_matrix = np.array(cv2.estimateRigidTransform(edges[template_band],
                                                 edges[i],
                                                 fullAffine=False), dtype=np.float32)
        
        if ECC_criterion == True:
            # Optimize using ECC criterion and default settings
            warp_matrix = cv2.findTransformECC(edges[template_band],
                                               edges[i],
                                               warpMatrix=warp_matrix)[1]
        # transform
        aligned = cv2.warpAffine(bands[i], 
                                 warp_matrix, 
                                 (width, height), 
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,  # otherwise the transformation goes the wrong way
                                 borderMode=cv2.BORDER_CONSTANT)
        
        # add to color image                             
        out[:, :, i] = aligned
    
    return(img_as_ubyte(out))

 
###############################################################################################
#########                                                                        ##############
#########                           Preprocessing Filters                        ##############
#########                                                                        ##############
###############################################################################################
def preprocessing_filters(image,
                          blur_params=None, 
                          missing_band_params=None, 
                          low_contrast_params=None,
                          center=True):
                         # brightfield_params=None, 
                          #registration_params=None,
                          #bilateral_filter_params=None):
    """
    Meta function for preprocessing images.
    
    Parameters
    ----------
    image : ndarray
        input rgb image
    blur_band : int
        band of rgb to check for blur
    blur_params : dict or `None`
        parameters for `pyroots.detect_blur`
    missing_band_params : dict or `None`
        parameters for `pyroots.detect_missing_bands`
    low_contrast_params : dict or `None`
        parameters for `skimage.exposure.is_low_contrast`
    center : bool
        Take middle 25% of an image for blur detection?
    
    Returns
    -------
    bool - should the image be pre-processed? Must pass all criteria given.
    
    """

    try:
        if center is True:
            blur = detect_motion_blur(_center_image(image), **blur_params)
        else:
            blur = detect_motion_blur(image, **blur_params)    
    except:
        blur = True
        if blur_params is not None:
            warn("Skipping motion blur check", UserWarning)
        pass
        
    try:
        bands = detect_missing_bands(image, **missing_band_params)
    except:
        bands = True
        if missing_band_params is not None:
            warn("Skipping missing band check", UserWarning)
        pass
        
    try:
        contrast = ~exposure.is_low_contrast(filters.gaussian(image, sigma=10, multichannel=True), **low_contrast_params)
    except:
        contrast = True
        if low_contrast_params is not None:
            warn("Skipping low contrast check", UserWarning)
        pass
    
    return(blur * bands * contrast)


###############################################################################################
#########                                                                        ##############
#########                            Preprocessing Actions                       ##############
#########                                                                        ##############
###############################################################################################
def preprocessing_actions(image,
                          brightfield,
                          brightfield_params=None, 
                          registration_params=None,
                          smoothing_params=None,
                          count_warnings=True):
    """
    Combines preprocessing functions into a convenience function.
    
    Parameters
    ----------
    image : ndarray
        input rgb image
    brightfield_params : dict or `None`
        parameters for `pyroots.correct_brightfield`
    registration_params : dict or `None`
        parameters for `pyroots.register_bands`
    bilateral_filter_params : dict or `None`
        parameters for `cv2.bilateralFilter`, which smooths the image while preserving edges.
    count_warnings : bool
        also return a flag counting number of warnings encountered?
    
    Returns
    -------
        1. ndarray of `image.shape` after running through functions listed.
        2. A marker for flagging errors (if `warn=True`)

    """
    out = image.copy()
    warning_flag = 0
    
    try:
        out = correct_brightfield(out, brightfield, **brightfield_params)
    except:
        if brightfield_params is not None:
            warning_flag += 1
            warn("Skipping brightfield correction", UserWarning)
        pass
    
    try:
        out = cv2.bilateralFilter(out, -1, **smoothing_params)
    except:
        if smoothing_params is not None:
            warning_flag += 1
            warn("Skipping bilateral filter", UserWarning)
        pass
    
    try:
        out = register_bands(out, **registration_params)
    except:
        if registration_params is not None:
            warning_flag += 1
            warn("Skipping band registration", UserWarning)
        pass
        
    if count_warnings == True:
        out = [out, warning_flag]
    
    return(out)

