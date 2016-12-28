"""
Created on Fri May  6 11:24:24 2016

@author: pme

Contents:
multi_image_plot
random_blobs
"""

from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np

def multi_image_plot(images, titles, 
					 color_map="gray", axis="off", 
					 titlesize=16, interpolation="None"):
	"""
	Function to plot multiple images along a single row for easy comparisons. 
	Requires matplotlib.pyplot

	Parameters
	----------
	images : list of images to plot in line
	titles : list of titles (strings) for each image
	color_map : default color map for 1d images (string)
	axis : show pixel location axis? (boolean)
	titlesize : size of titles (integer)
	interpolation : interpolation method for rendering
	
	Returns
	-------
	An array of images generated using matplotlib.pyplot.show()

	"""
	from matplotlib import pyplot as plt
	
	n = len(images)
	
	plt.figure(figsize=(3*n, 4))
	
	for k in range(1, n+1):
		plt.subplot(1, n, k)
		plt.imshow(images[k-1], cmap=color_map, interpolation = interpolation)
		plt.axis(axis)
		plt.title(titles[k-1], size=titlesize)
	
	plt.subplots_adjust(wspace=0.02, hspace=0.02, 
						top=0.9, bottom=0, left=0, right=1)

	return plt.show()
	

def random_blobs(n=100, dims=256, seed=1, size=0.25, noise=True):

	"""
	Function to create a square image with blobs formed around randomly placed
	points. The image can include random noise to make everything blurry
	for testing methods. Requires numpy and scipy.ndimage.
	
	Parameters
	----------
	n : set the number of points (integer)
	dims : length of each side of the square (integer)
	seed : set random number seed (integer)
	size : set relative size of blobs around each point (float)
	noise : add noise to the image

	Returns
	-------
	An dims*dims array of blobs as either boolean or float
	
	"""


	im = np.zeros((dims, dims))
	np.random.seed(seed)
	points = (dims * np.random.random((2, n))).astype(np.int) #blob locations
	im[(points[0]), (points[1])] = 1
	im = ndimage.gaussian_filter(im, sigma=float(size) * dims / (n)) #blob size
	mask = (im > im.mean()).astype(np.float) #make hard lines around blobs
	
	if noise is True:
		mask += 0.1*im
		img = mask + 0.2 * np.random.randn(*mask.shape)	#matrix of the shape of mask
		img += abs(img.min())
		img = img / img.max()
	else:
		img = mask
	return img
