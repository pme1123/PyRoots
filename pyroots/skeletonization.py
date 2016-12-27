# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:10:25 2016

@author: pme

Contents:
- _axis_length
- skeletonize_with_distance

"""

import pandas as pd
import numpy as np
from scipy import ndimage
from skimage import morphology



def _axis_length(image, labels=None, random=True, m=0.5):
	"""
	Calculate the length of each pixel in a non-straight line, based on
	results in Kimura et al. (1999) [1]. This is a modified version of 
	skimage.morphology.perimeter [2].
	
	Parameters
	----------
	image : skeletonized array. Binary image (required)
	labels : array of object labels. Integer values, for example from 
	ndimage.label(). Default=None
	random : boolean. Are the segments randomly oriented, or are they well 
	ordered? Default=True. 
	m : float [0,1]. For random=False, otherwise the algorithm will 
	underestimate length. Default=0.5 This should work for most situations. 
	See [1]. 
		
		
	Returns
	-------
	length : float
		If no labels are included, the function returns the total length of
		all objects in binary image.
		or
	length : ndarray
		If a label array is included, the function returns an array of lengths
		for each labelled segment.
		
	See Also
	--------
	skimage.morphology.filter
	
	References
	----------
	.. [1] Kimura K, Kikuchi S, Yamasaki S. 1999. Accurate root length 
		   measurement by image analysis. Plant and Soil 216: 117–127.
	.. [2] K. Benkrid, D. Crookes. Design and FPGA Implementation of
		   a Perimeter Estimator. The Queen's University of Belfast.
		   http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
	
	"""
		
	image = image.astype(np.uint8)

	#Define the connectivity list
	if random is True:
		pixel_weight = np.array( 
			#Generated from below, to weight the sum values of connectivity 
			#from the kernel passed to ndi.convolve
			[0.   ,  0.5  ,  0.   ,  0.474,  0.   ,  0.948,  0.   ,  1.422,
			 0.   ,  1.896,  0.   ,  0.67 ,  0.   ,  1.144,  0.   ,  1.618,
			 0.   ,  2.092,  0.   ,  2.566,  0.   ,  1.341,  0.   ,  1.815,
			 0.   ,  2.289,  0.   ,  2.763,  0.   ,  3.237,  0.   ,  2.011,
			 0.   ,  2.485,  0.   ,  2.959,  0.   ,  3.433,  0.   ,  3.907,
			 0.   ,  2.681,  0.   ,  3.155,  0.   ,  3.629,  0.   ,  4.103,
			 0.   ,  4.577])

#	   weight_seq = np.arange(0, 50, step=1, 
#							  dtype = "float")	 
#	   horiz_vert = (weight_seq % 10) // 2	#Count horiz, vertical connects
#	   diag = weight_seq // 10				#Count diagonal connections
#	   pixel_weight = 0.5*(horiz_vert + diag*(2**0.5))
#	   pixel_weight = pixel_weight*0.948	  #Correction factor assuming
#												 #random orientations of lines 
#	   pixel_weight[np.arange(0, 50, 2)] = 0  #Only odd numbers are on skeleton
#	   pixel_weight[1] = 0.5				  #Account for lone pixels
#			  
												  
	else:	#This weighting systematically overestimates (slightly), but the 
			 #error should be smaller for well-oriented roots. See Kimura et al.
		 	 #(1999)
		if m is 0.5:
			pixel_weight = np.array(
			#Generated from the else: statement below for m = 0.5
			[0.   ,  0.5  ,  0.   ,  0.5  ,  0.   ,  1.   ,  0.   ,  1.5  ,
			 0.   ,  2.   ,  0.   ,  0.707,  0.   ,  1.151,  0.   ,  1.618,
			 0.   ,  2.096,  0.   ,  2.581,  0.   ,  1.414,  0.   ,  1.851,
			 0.   ,  2.303,  0.   ,  2.766,  0.   ,  3.236,  0.   ,  2.121,
			 0.   ,  2.555,  0.   ,  3.   ,  0.   ,  3.454,  0.   ,  3.915,
			 0.   ,  2.828,  0.   ,  3.26 ,  0.   ,  3.702,  0.   ,  4.15 ,
			 0.   ,  4.606])
		else:
			weight_seq = np.arange(0, 50, step=1, dtype="float")
			orth = (weight_seq % 10) // 2  #count orthogonal links
			diag = weight_seq // 10		#count diagonal links
			pixel_weight = 0.5 * ((diag**2 + (diag + orth*m)**2)**0.5 + orth*(1-m)) 
			pixel_weight[np.arange(0, 50, 2)] = 0  #Only odd numbers are on skeleton
			pixel_weight[1] = 0.5				  #Account for lone pixels

	#Run the connectivity kernel	
	kernel = np.array([[10, 2, 10],
					   [ 2, 1,  2],
					   [10, 2, 10]])

	kernel_out = ndimage.convolve(image, kernel,
								  mode='constant', cval=0
								  )
	
	#convert kernel_out to pixel length, for diagnostics
	dims = kernel_out.shape
	pixel_length = np.ones(dims)	#most should become zeros
	for i in range(dims[0]):		#x dimension
		for j in range(dims[1]):	#y dimension
			pixel_length[i,j] = pixel_weight[kernel_out[i,j]] 
				#match the value to that in the pixel_weight list
			
	if labels is None: 
		
		# From original code - skimage.morphology.perimeter():
		# "You can also write
		# return perimeter_weights[perimeter_image].sum()
		# but that was measured as taking much longer than bincount + np.dot 
		# (5x as much time)"
		weight_bins = np.bincount(kernel_out.ravel(), minlength=50)
		length_out = np.dot(weight_bins, pixel_weight)
		return(weight_bins, 
			   length_out)
	else:
		dims = kernel_out.shape
		pixel_length = np.zeros(dims)	#most should stay zeros
		for i in range(dims[0]):		#x dimension iterate
			for j in range(dims[1]):	#y dimension iterate
				pixel_length[i,j] = pixel_weight[kernel_out[i,j]] 
		length_out = ndimage.sum(pixel_length, 
								 labels=labels, 
								 index=range(labels.max() + 1)[1:]
								 #might want to ignore index 0 
								 #for large images.
								 )
		return(pixel_length,
			   length_out)
			   
			   
def skeleton_with_distance(img, random=True, m=0.5):
	"""
	Created on Thu 21 Jul 2016 03:27:52 PM CDT 
	
	@author: pme

	Calls morphology.medial_axis(), then calculates the length of each axis.

	Parameters
	--------
	img: boolean ndarray
	random: Are the segments randomly oriented? Boolean, default = True. 
		See source code for pyroots.axis_length. 
	m: Pamameter for pyroots.axis_length. Default = 0.5. See source code.

	Returns
	--------
	A list containing:
		An ndarray of length for each pixel
		An ndarray of diameter at each pixel
		A list of vectors (mean length, diameter) for each object. The 
			first index is open space and therefore 0 (for consistency with 
			ndimage.label() behavior). 
	"""
	
	labels, labels_ls = ndimage.label(img)
	skel, dist = morphology.medial_axis(img, return_distance=True)
	dist_img = skel*2*dist #2x because medial axis distance is radius
	label_skel = skel*labels
	
	width_list = ndimage.mean(dist_img, 
							  label_skel, 
							  index=range(labels_ls+1)[1:]) #ignore empty space
	
	length_img, length_list = _axis_length(skel, labels, random=random, m=m)
	geom_df = pd.DataFrame({"Length" : np.insert(length_list, 0, 0), #to re-add the label index, 0
				  "Diameter" : np.insert(width_list, 0, 0)
				}) 
	

#	lw_ratio = length / (width+0.0000001)
#	lw_filter_img = 
	
	
	return(length_img, dist_img, geom_df)
