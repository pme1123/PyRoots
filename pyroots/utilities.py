"""
Created on Fri May  6 11:24:24 2016

@author: pme

Contents:
_zoom
band_viewer
multi_image_plot
random_blobs
tiff_splitter
img_rescaler
"""

from matplotlib import pyplot as plt
from scipy import ndimage
import numpy as np
from skimage import io, color
import os
from multiprocessing.dummy import Pool
import pyroots as pr
from skimage import io, img_as_ubyte
import tqdm
from multiprocessing import Pool
from time import sleep

def _zoom(image, xmin, xmax, ymin, ymax, set_scale=False):
    """
    Subset an array to the bounding box suggested by the titles
    If `set_scale`, then the first two pixels of the slice are set to the
    max and min of the entire image.
    """
    out = image[ymin:ymax, xmin:xmax]
    if set_scale:
        if len(image.shape) == 3:
            out[0, 0, 0:3] = [np.max(image[:, :, i]) for i in range(image.shape[2])]
            out[1, 1, 0:3] = [np.min(image[:, :, i]) for i in range(image.shape[2])]
        else:
            out[0, 0] = np.max(image)
            out[1, 1] = np.min(image)
    
    return(out)

def band_viewer(img, colorspace, zoom_coords = None, return_bands=False):
    """
    Utility function to look at the separate bands of multiple colorspace versions.
    
    Parameters
    ----------
    img : array
        RGB image, such as imported by ``skimage.io.imread``
    colorspace : str
        Colorspace to which to convert. Must be finish one the ``skimage.color.rgb2*``
        functions.
    zoom_coords : list
        List of four integers denoting the start and end of an area of interest to view more closely, or none
    return_bands : bool
        Do you want to return the bands of the colorspace in an object?
    
    Returns
    -------
    Plots the bands of the given colorspace using ``pyroots.multi_image_plot``. If
    ``return_bands`` is ``True``, returns these bands in a list.
    
    See Also
    --------
    ``skimage.color``, ``pyroots.multi_image_split``
    
    """
    if zoom_coords is None:
        zoom_coords = {'xmin' : 0,
                       'xmax' : img.shape[1],
                       'ymin' : 0,
                       'ymax' : img.shape[0]}
        
    elif isinstance(zoom_coords, list) is True:
        zoom_coords = {'xmin' : zoom_coords[0],
                       'xmax' : zoom_coords[1],
                       'ymin' : zoom_coords[2],
                       'ymax' : zoom_coords[3]}
    
    elif isinstance(zoom_coords, dict) is False:
        raise "Zoom Coordinates Issue"
    
    #image is rgb
    if colorspace is not "rgb":
        img = getattr(color, 'rgb2' + colorspace)(img)
    bands = pr.img_split(img)
    
    
    if return_bands is True:
        return (bands)

    else:
        pr.multi_image_plot([pr._zoom(i, **zoom_coords) for i in bands], 
                            [colorspace[0], colorspace[1], colorspace[2]])
                            

    
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
        img = mask + 0.2 * np.random.randn(*mask.shape)    #matrix of the shape of mask
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


def img_rescaler(dir_in, extension_in, threads=1):
    """ 
    Import an image, rescale it to normal UBYTE (0-255, 8 bit) range, and re-save it.
    
    """

    dir_out = os.path.join(dir_in, "rescaled")
    
    total_files = 0
    for path, folder, filename in os.walk(dir_in):
        if dir_out not in path:
            for f in filename:
                if f.endswith(extension_in):
                    total_files += 1
    print("\nYou have {} images to analyze".format(total_files))
    
    for path, folder, filename in os.walk(dir_in):
        if dir_out not in path:   # Don't run in the output directory.

            # Make directory for saving objects
            subpath = path[len(dir_in)+1:]
            if not os.path.exists(os.path.join(dir_out, subpath)):
                os.mkdir(os.path.join(dir_out, subpath))

            # What we'll do:
            global _core_fn  # bad form for Pool.map() compatibility
            def _core_fn(filename):
                if filename.endswith(extension_in):
                    # count progress.

                    path_in = os.path.join(path, filename)
                    subpath_in = os.path.join(subpath, filename) # for printing purposes
                    path_out = os.path.join(dir_out, subpath, filename)

                    if os.path.exists(path_out): #skip
                        print("\nALREADY ANALYZED: {}. Skipping...\n".format(subpath_in))

                    else: #(try to) do it
                        try:
                            img = io.imread(path_in)  # load image
                            img = img_as_ubyte(img / np.max(img))
                            io.imsave(path_out, img)
                        except:
                            print("Couldn't analyze {}".format(subpath_in))
                return()
            
            # run it
            sleep(1)  # to give everything time to  load
            thread_pool = Pool(threads)
            # Work on _core_fn (and give progressbar)
            tqdm.tqdm(thread_pool.imap_unordered(_core_fn,
                                                 filename,
                                                 chunksize=1),
                      total=total_files)
            # finish
            thread_pool.close()
            thread_pool.join()
    return()
    
