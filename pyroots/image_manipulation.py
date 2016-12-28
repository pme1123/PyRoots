"""
Created on Fri May  6 14:55:54 2016

@author: pme


Contents:
- img_split
- circle_mask

"""

import numpy as np

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


def circle_mask(img, param=None):
	"""
	Create an elliptical mask to limit analysis to the center of an image.
	Useful for microscope images that are blurry at the edges, for example.

	Parameters
	----------
	img : array
		image upon which to draw the mask
	param : int
		list of x and y dimensions of the ellipse. Defaults to an ellipse
		spanning the image.

	Returns
	-------
	A boolean ndarray where the central circle is 0 and the edges are 1.

	"""

	if param is None:
		param_x = img.shape[0]//2
		param_y = img.shape[1]//2
	else:
		param_x, param_y = param[0:1]

	dim_x, dim_y = img.shape[0]//2, img.shape[1]//2	   
   
	x, y = np.ogrid[-param_x:param_x, -param_y:param_y] #makes an open grid of 
	mask = x**2 + y**2 >= param_x**2 #>= removes the outside edges. 
	
	return(~mask)
