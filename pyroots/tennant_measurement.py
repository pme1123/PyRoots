#! /bin/python3

"""
Author: Patrick Ewing
Date: May 23, 2018

Functions for tenant segmentation. Mostly for validation purposes to compare
manual length estimates with pyroots segmentation. 

Contents
--------
    tennant_on_segmented
    draw_fishnet
"""

from scipy import ndimage
import numpy as np
from skimage import color, img_as_ubyte

#####################################################
#####################################################
#######                                        ######
#######             File Subsampler            ######
#######                                        ######
#####################################################
#####################################################

def tennant_on_segmented(binary_image, grid_size):
    """
    Estimates the length of objects in `binary_image` using the method of Tennant (1976),
    the line-intercept method. The image should be the output of one of the segmentation
    methods, because those only inlcude objects that pass post-medial axis filtering like 
    length:width filters.
    
    Parameters
    ----------
    binary_image : array
        A segmented image. Objects are `True`.
    grid_size : int
        length of fishnet side in pixels
        
    Returns
    -------
    The number of crosses (in pixels) = number of objects the intersection of the grid and
    the binary image.
    """
    
    binary_image = binary_image > 0
    
    # make grid of same size
    grid = np.zeros(binary_image.shape)
    grid = draw_fishnet(grid,
                        size=grid_size,
                        grid_color=(100, 100, 100),
                        weight=1
                       )
    grid = grid[:, :, 0] > 0
    
    obj = grid*binary_image
    
    crosses = ndimage.label(obj)[1]
    
    return(crosses)
    
#####################################################
#####################################################
#######                                        ######
#######              Draw Fishnet              ######
#######                                        ######
#####################################################
#####################################################
def draw_fishnet(image_in,
                 size=100,
                 grid_color = (200, 0, 0),
                 weight = 1):

    """
    Draw a square mesh grid over an image. The size of the grid is in pixels.
    Color is a 3-value list giving RGB values.

    Parameters
    ----------
    image_in : ndarray
        Probably 3D. If not, is converted to 3D (i.e. grayscale to color).
    size : int
        Pixel length of one edge of the square mesh.
    color : int
        3 values specifying the color of the grid in 8-bit RGB space. Default is red.
    weight : int
        pixels wide of the lines. If is float, rounds down to int.
    
    Returns
    -------
    A 3D array - the image with the grid. 
    
    """
    image = image_in.copy()

    
    # convert gray image to rgb or otherwise test for compatibility
    if len(image.shape) == 2:
        image = color.gray2rgb(image)
    elif len(image.shape) != 3 or image.shape[2] != 3:
        msg = 'Image must be 1D or 3D'
        raise ValueError(msg)
        
    if len(grid_color) != 3:
        msg = '`grid_Color` must be a 3-value list or tuple (give RGB values)'
        raise ValueError(msg)
    
    image = img_as_ubyte(image)
    
    dimy, dimx = image.shape[0:2]
    
    def _grid_coords(a, b):
        """
        List of points spaced b from 0:a, starting at b/2
        """
        new = round(b/2)  # offset half a grid
        vals = []
        while new < a:
            vals.append(new)
            new += b
        return vals
    
    y_vals = _grid_coords(dimy, size)
    x_vals = _grid_coords(dimx, size)
    
    for i in range(1, weight):
        y_vals += [j + i for j in y_vals]
        x_vals += [j + i for j in x_vals]
    
    grid_color = [int(i) for i in grid_color]
   
    for i in y_vals:
        for j in range(3):
            image[i, :, j] = grid_color[j]
            
    for i in x_vals:
        for j in range(3):
            image[:, i, j] = grid_color[j] 
    
    image = img_as_ubyte(image)
            
    return(image)
