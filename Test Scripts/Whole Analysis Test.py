# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 15:25:27 2016

@author: patrick

Whole Analysis Test
"""

import pyroots as pr
from skimage import io
from matplotlib import pyplot

dir_in = "/home/patrick/Cloud/Codes/pyroots/pyroots/sample_images/"
file_in = "hyphae_500x500.tif"
image = io.imread(dir_in + file_in)[:,:,0:3]

load "/home/patrick/Cloud/Codes/pyroots/pyroots/analysis_parameters.py"

test = pr.pyroots_analysis(image, file_in, 
                           colorspace, analysis_bands, 
                           threshold_params, 
                           filtering_params, light_on_dark, 
                           mask=None,
                           diameter_bins=None, optimize=True)

test

image_loop("/home/patrick/Cloud/Codes/pyroots/pyroots/sample_images", ".tif", 
           "/home/patrick/Cloud/Codes/pyroots/pyroots/analysis_parameters.py", 
           save_images=True)
