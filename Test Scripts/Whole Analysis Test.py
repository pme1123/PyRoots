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

