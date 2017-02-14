"""
Functions to sort and prepare raw input images for analysis with pyroots functions. 

Contents:
- detect_blur
- detect_missing_bands
- correct_brightfield
- register_bands
- preprocessing_filters
"""
import numpy as np
from skimage import filters, img_as_ubyte, exposure
from pyroots import img_split
import cv2


###############################################################################################
#########                                                                        ##############
#########                               Blur Detector                            ##############
#########                                                                        ##############
###############################################################################################

def detect_blur(image, threshold=10, direction='both'):
    """
    Detect if an image is blurry due to movement or extremely poor focus, and therefore unreliable
    to analyze. Uses the variance of the prewitt edge detectors in the horizontal and vertical directions.
    
    Parameters
    ----------
    image : ndarray
        grayscale image. 
    threshold : float
        Minimum variance of edge strength. Default 10.
    direction : str
        Apply the threshold in the `'horizontal'`, `'vertical'`, or `'both'` (default) directions? 
        
    Returns
    -------
    bool - does the image meet requirements?
    
    Notes
    -----
    `direction = 'horizontal'` applies the _vertical_ prewitt operator to test blurriness in the
    _horizontal_ (x) direction. 'vertical' is the opposite.
    
    """
    if direction not in ['horizontal', 'vertical', 'both']:
        raise NameError("'direction' must be 'horizontal', 'vertical', or 'both'")
    
    if direction != "vertical":
        horiz = img_as_ubyte(filters.prewitt_v(image)).var() > threshold
    else:
        horiz = True
    
    if direction != "horizontal":
        vert = img_as_ubyte(filters.prewitt_h(image)).var() > threshold
    else:
        vert = True
    
    out = horiz * vert
    
    return(out)


###############################################################################################
#########                                                                        ##############
#########                           Missing Band Detector                        ##############
#########                                                                        ##############
###############################################################################################

def detect_missing_bands(image, percentile=90, threshold=0.0):
    """
    Detecs whether an image is missing a band. This happens with some camera processing software.
    
    Parameters
    ----------
    image : ndarray
        RGB image
    percentile : float
        percentile of values in each band to test against the threshold. Default 90.
    threshold : float
        floor of values at each band to pass. Default 0.0
    
    Returns
    -------
    bool - Does the image pass criteria?
    """
    tests = img_split(image)
    tests = [np.percentile(i, percentile) > threshold for i in tests]
    
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
def register_bands(image, template_band=1):
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
    
    Returns
    -------
    An ndarray of `image.size`
    
    Notes
    -----
    Uses `skimage.filters.scharr` to find edges in each band, then finds and
    applies an affine transformation to register the images using 
    `cv2.estimateRigidTransform` and `cv2.warpAffine`.
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
        # calculate transformation
        warp_matrix = cv2.estimateRigidTransform(edges[template_band],
                                                 edges[i],
                                                 fullAffine=True)
        # transform
        aligned = cv2.warpAffine(bands[i], 
                                 warp_matrix, 
                                 (width, height), 
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,  # otherwise the transformation goes the wrong way
                                 borderMode=cv2.BORDER_CONSTANT)
        
        # add to color image                             
        out[:, :, i] = aligned
    
    return(out)

 
###############################################################################################
#########                                                                        ##############
#########                           Preprocessing Filters                        ##############
#########                                                                        ##############
###############################################################################################
def preprocessing_filters(image, 
                          blur_band=0,
                          blur_params=None, 
                          missing_band_params=None, 
                          low_contrast_params=None):
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
    brightfield_params : dict or `None`
        parameters for `pyroots.correct_brightfield`
    registration_params : dict or `None`
        parameters for `pyroots.register_bands`
    bilateral_filter_params : dict or `None`
        parameters for `cv2.bilateralFilter`, which smooths the image while preserving edges.
    
    Returns
    -------
    bool - should the image be pre-processed? Must pass all criteria given.
    
    """
    
    blur = detect_blur(image[:, :, blur_band], **blur_params)
    bands = detect_missing_bands(image, **missing_band_params)
    contrast = ~exposure.is_low_contrast(filters.gaussian(image, sigma=10), **low_contrast_params)
    
    return(blur * bands * contrast)


###############################################################################################
#########                                                                        ##############
#########                            Preprocessing Actions                       ##############
#########                                                                        ##############
###############################################################################################
def preprocessing_actions(image,
               brightfield_params=None, 
               registration_params=None,
               bilateral_filter_params=None):
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
    
    Returns
    -------
    ndarray of `image.shape` after running through functions listed.
    
    """
    out = image.copy()
    
    try:
        out = register_bands(out, **registration_params)
    except:
        pass
        
    try:
        out = correct_brightfield(out, **brightfield_params)
    except:
        pass
    
    try:
        out = cv2.bilateralFilter(out, **bilateral_filter_params)
    except:
        pass
    
    return(out)

