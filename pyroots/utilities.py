"""
Created on Fri May  6 11:24:24 2016

@author: pme

Contents:
multi_image_plot
random_blobs
tiff_splitter
"""

from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np

# for tiff splitter
from skimage import io
import os
from multiprocessing.dummy import Pool

def multi_image_plot(images, titles, 
					 color_map="gray", axis="off", 
					 titlesize=16, interpolation="None"):
	"""
	Wrapper function for ``pyplot.imshow`` to plot multiple images along a 
	single row for easy comparisons. Requires ``matplotlib.pyplot``

	Parameters
	----------
	images : list
		Image arrays to plot
	titles : list 
		Titles (strings) for each image
	color_map : string
		Color map for 1d images. 3d images plot normally
	axis : bool
		Show pixel location axis?
	titlesize : int
		Size of titles 
	interpolation : str
		Interpolation method for rendering
	
	Returns
	-------
	An array of images generated using ``matplotlib.pyplot.show()``

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
	for testing methods. Requires ``numpy`` and ``scipy.ndimage``.
	
	Parameters
	----------
	n : int
		set the number of points
	dims : int
		length of each side of the square
	seed : int
		set random number seed
	size : float
		set relative size of blobs around each point
	noise : bool
		add noise to the image

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
	

def tiff_splitter(directory_in, extension=".tif", threads=1):
    """
    Silly function to load a tiff file and resave it. This is critical if the
    tiff is a multi-page tiff (ex. images, thumbnails) and you want to
    pre-process it with GIMP, for example. The output is the first page of the 
    tiff.
    
    Parameters
    ----------
    directory_in : str
        Directory where the images are
    extension : str
        Image extension. Defaults to ```".tif"```, but could be anything.
    threads : int
        For multithreading. This can be a stupidly slow function.
        
    Returns
    -------
    Creates a directory called "split_images" in ```directory_in``` that has
    copies of the same images, but without thumbnails.
    
    """
    
    #core function to map across threads
    def _load_unload_image(file_in):
            
        if file_in.endswith(extension):                   
            path_in = subdir + os.sep + file_in  # what's the image called and where is it?
            path_out = directory_out + sub_path + os.sep + file_in
                
            #Import and export image
            io.imsave(path_out, io.imread(path_in))
            print("Split: " + ".." + sub_path + os.sep + file_in)
            
        else:
            print("Skip: " + ".." + sub_path + os.sep + file_in)
        
    # housekeeping
    directory_out = directory_in + os.sep + "split_images"
    if not os.path.exists(directory_out):
        os.mkdir(directory_out)
    
    # initiate threads
    threadpool = Pool(threads)
    
    for subdir, dirs, files in os.walk(directory_in):
        sub_path = subdir[len(directory_in): ]
        
        if not "split_images" in subdir:
            if not os.path.exists(directory_out + subdir[len(directory_in): ]):
                os.mkdir(directory_out + subdir[len(directory_in): ])
                
            threadpool.map(_load_unload_image, files)

    # end threads
    threadpool.close()
    threadpool.join() 
                  
    return("Done")
    
    
    
