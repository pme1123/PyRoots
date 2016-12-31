"""
Created on Fri May  6 14:55:54 2016

@author: pme


Contents:
- img_split
- circle_mask

"""

import numpy as np
from skimage import draw

def img_split(img):

	"""
	Split a multispectral image into its bands. 
	
	Parameters
	----------
	img : array
		A 3- or 4-band image or array

	Returns
	-------
	Returns a list where each section is a separate band
	in the array. ex. RGB image returns [R, G, B].

	"""
	bands = img.shape[2]
	if bands is 1:
		return "Image already is 1D. Why would you split it?"

	band1 = img[:, :, 0]
	band2 = img[:, :, 1]
	band3 = img[:, :, 2]
	if bands is 4:
		band4 = img[:, :, 4]
		return(band1, band2, band3, band4)
	return(band1, band2, band3)


def ellipse_mask(img, percentage_x, percentage_y, offset_x, offset_y, rotation = 0):
    """
    Convenience wrapper for ``skimage.draw.ellipse``. Draws an ellipse at (``center`` + 
    ``offset``) of ``img`` with major and minor axes as a percentage of ``img.shape``. 
    
    Parameters
    ----------
    img : array
        Image array that you want to draw a mask on
    
    percentage_x : float
        What percentage of the x-dimension of ``img`` do you want the x-axis to cover? Can
        be greater than zero
    percentage_y : float
    offset_x : int
        From the middle, how many pixels in the x direction do you want the center of the
        ellipse to be? Positive is left.
    offset_y : int
        Positive is up.
    rotation : float
        In [-pi, pi]. See ``skimage.draw.ellipse''
    
    Returns
    -------
    A binary array with pixels in an ellipse.
    
    References
    ----------
    See source and examples for ``skimage.draw.ellipse``
    
    """
    
    x_rad = np.floor((img.shape[0]/2) * (percentage_x/100))
    y_rad = np.floor((img.shape[1]/2) * (percentage_y/100))
    
    x_center = img.shape[0]//2 + offset_x
    y_center = img.shape[1]//2 - offset_y
    
    mask = np.zeros(img.shape)
    [x, y] = draw.ellipse(y_center, x_center, y_rad, x_rad, shape = img.shape)
    mask[x, y] = 1
    
    return(mask)
