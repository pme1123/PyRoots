"""
Created on Tues Dec 27th 2016

author: pme

This is a sample parameters file for running the pyroots_analysis function. Load it by running source("path_to_this_file") in python3.

Note the dictionary format. Names of items in the dictionary correspond to arguments in the identified function.
See documentation of those functions for details, plus more parameters.

The parameters here work decently for blue objects on light backgrounds. It directs the analysis to run on the b band
of the LAB colorspace and identifies objects that have a length:width ratio > 5 and are generally narrower than 10 px wide
as real.

TODO: Make a jupyter notebook to build this parameters script and an analysis function interactively. 

"""

# Where is your file?
import_params = {"directory" : "/home/patrick/Cloud/Codes/pyroots/pyroots/sample_images/", 
                 "filename"  : 'hyphae_500x500.tif'}

# Which colorspace do you want to use? Options include HSV, RGB, LAB, CMY. 
colorspace = 'lab'

# Are the roots light colored on a dark background?
light_on_dark = False

# which bands of the colorspace do you want? ex. in RGB, R = 0, G = 1, B = 2.
analysis_bands = [2]  # If you want to use more than one band, edit the source code to identify how to combine them into a single, binary image.

# Do you want to analyze a consistent part of the image?
mask_params = [{'form' : 'ellipse'},  # What ort of mask? Options include: None, 'ellipse'
        	   {'param' : None}]      # List of length = 2 showing the major and minor axes of the ellipse, or None to center a circle.

# skimage.filters.threshold_adaptive(). Works on individual analysis bands.
threshold_params = {'block_size' : 191,          # Length of square sides to use as a thresholding neighborhood. Must be odd. 
                    'offset'     : 5,            # Add this value to param to set as threshold. Larger is more stringent. 
                    'param'      : ((191-1)/6)}  # Standard deviations from mean to use as the threshold for each locale. Default is(block_size-1)/6. 

# Filter noise and clean up the binary (threshold) image.
         ## pyroots.dirt_removal(). Works on binary image.
filtering_params = [{'param' : 5},  # Number of 'standard deviations' from 0 that marks the limit of noise objects. Larger removes more objects. Default = 5.

		 ## pyroots.noise_removal(). Works on binary image. 
            {'structure_1' : "Default",  # Define distance for separating loosely connected objects. "Default": manhattan = 1.
		    'structure_2' : "Default"},  # Define distance for smoothing with median filter. Default": maximum = 2. 

		 ## pyroots.length_width_filter(). Works on geometry dataframe.
            {'threshold' : 5},  # Larger is more stringent. Recommended = 5. Default = 15.
		
	     ## pyroots.diameter_filter(). Works on diameter skeleton.
            {'max_diameter'   : 10,     # Pixels. Remove diameter pixels larger than max_diameter. Arbitrarily large to skip. Default = 10
             'min_diameter'   : -1,     # Pixels. Removes diameter skeleton pixels smaller than this. Negative to skip. Default = -1
             'max_percentile' : 80,     # Percent. Removes entire objects where (100-%) pixels are wider than max_diameter. 100 skips. Default = 80
             'min_percentile' : None}]  # Percent. Removes entire objects where (100-%) pixels are narrower than min_diameter. Default = None to skip.

# pyroots.bin_by_diameter(). Works on diameter and length skeleton. 
diameter_bins = None  # Specify break points in a float list [#, #, #] if you want to measure the length of specific diameter classes. In pixels. Default = None.
