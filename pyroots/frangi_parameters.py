#! /bin/python3/

"""
Parameters for frangi filtering

Change to `None` to ignore. Save a copy to change. Works with ``pyroots.frangi_approach``.
"""

# Pick colorspace and bands
colors = {'colorspace' : 'rgb',  # must be reachable with `skimage.color.rgb2***()`
          'band'       : 0}      # RGB: 0 = red, 1 = green, 2 = blue. 

# Frangi vessel enhancement parameters. Passes to `skimage.filters.frangi`
frangi_args = {'scale_range'  : (2, 8),  # sigmas for running the filter. Wider enhances wider objects; narrow enhances narrow objects.
               'scale_step'   : 1,       # step between sigmas in the scale range.
               'beta1'        : 0.99,    # correction [0,1] toward linear vs ring-like. Want close to 1.
               'beta2'        : 0.05,    # sensitivity to low-contrast areas
               'black_ridges' : True}    # dark objects on light background?

# Adaptive threshold frangi vessels. Passes to `skimage.filters.threshold_adaptive`
threshold_args = {'block_size' : 29,    # small enough to respond strongly to the local neighborhood, large enough to avoid hollowing objects
                  'offset'     : 0.09}  # high enough to separate touching objects, but low enough to maintain connectivity

               
# Filtering: Color (band value). First dict chooses colorspace and band. Second passes to pyroots.percentile_in_range
color_args_1 = [{'colorspace' : 'rgb',  # Colorspace from which to select band
                 'band'       : 2},     # Band of colorspace                
                {'low'     : 0.7,    # minimum of range for color value
                 'high'    : 0.85,   # maximum of range for color value
                 'percent' : 50}]    # minimum percent of all pixels that must fall with the range

color_args_2 = [{'colorspace' : 'hsv',  # Colorspace from which to select band
                 'band'       : 0},     # Band of colorspace                
                {'low'     : 0.5,    # minimum of range for color value
                 'high'    : 0.7,    # maximum of range for color value
                 'percent' : 60}]    # minimum percent of all pixels that must fall with the range
                 
# Filtering: Morphology. Passes to pyroots.morphology_filter.
    # Eccentricity is of the ratio of the distance between ellipse foci and the major axis length. As eccentricity --> 0, becomes a circle. 
    # Ellipse is an ellipse with equivalent moments to the convex hull
    # Solidity is the ratio of the object area to the convex hull area
    # Strict filters are stringent criteria, but the object only needs to pass one or the other.
    # Loose filters are more relaxed criteria, but the object must pass one. 
morphology_args = {'loose_eccentricity'  : 0.7,   # This is a floor.
                   'loose_solidity'      : 0.8,   # This is a ceiling.
                   'strict_eccentricity' : 0.95,  # Floor.
                   'strict_solidity'     : 0.3,   # Ceiling.
                   'min_length'          : 100,   # Floor. Of equivalent ellipse.
                   'min_size'            : 300}   # Floor. Area in pixels. 

# Filtering: Hollowness. Passes to pyroots.hollow_filter.
# Hollowness is defined as the ratio of the medial axis length before and after binary closing. Perfectly closed objects have a theoretical ratio of 1. 
hollow_args = {'ratio'       : 1.3,  # this is a ceiling
               'fill_kernel' : 15}   # radius of disk, in pixels, passed to binary_closing
                 
# Fill small gaps and holes
hole_filling = {'selem'  : morphology.disk(7),  # passes to skimage.morphology.binary_closing. Larger is more aggressive.
                'min_size' : 300}               # passes to skimage.morphology.remove_small_holes. Minimum hole area in pixels.

# Summarize the image:
diameter_bins = None
