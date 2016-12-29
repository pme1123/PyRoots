# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:47:44 2016

@author: patrick

Test core functions in pyroots. Return a 
"""

import pyroots as pr
from skimage import io, filters
from matplotlib import pyplot

dir_in = "/home/patrick/Cloud/Codes/pyroots/pyroots/sample_images/"
file_in = "hyphae_500x500.tif"
image = io.imread(dir_in + file_in)[:,:,0:3]

#__all__ = ['pyroots_analysis', 'image_loop']

#pr.multi_image_plot([image, image], ['i','j'])

blobs = pr.random_blobs(n = 10, size = 1)
#pyplot.imshow(blobs, cmap = "BuPu")

bands = pr.img_split(image)
#pr.multi_image_plot(bands, ['r','g','b'])

mask = pr.circle_mask(bands[0])
#pyplot.imshow(mask, cmap = "gray")

bands = bands[1]
thresh = filters.threshold_adaptive(bands, 191, offset=5, param=6)
#pyplot.imshow(thresh)

#### dirt_removal ####
# statistical
dirt = pr.dirt_removal(~thresh, param = 5)
#pyplot.imshow(dirt, cmap = "gray")

# threshold
dirt1 = pr.dirt_removal(~thresh, method="threshold", param=100)
#pyplot.imshow(dirt1)

#### noise_removal ####
noise = pr.noise_removal(dirt)
#pyplot.imshow(noise, cmap = 'gray')

#### skeleton ####
skel_dict = pr.skeleton_with_distance(noise)
#pr.multi_image_plot([skel_dict['diameter'], skel_dict['length'], skel_dict['objects']],
#                    ['diam', 'length', 'objects'], color_map = "spectral")

diam_dict = pr.diameter_filter(skel_dict, max_diameter=8, max_percentile=80)
#pr.multi_image_plot([diam_dict['diameter'], diam_dict['length'], diam_dict['objects']],
#                    ['diam', 'length', 'objects'], color_map = "spectral")
#diam_dict["geometry"]

diam_dict1 = pr.diameter_filter(skel_dict, min_diameter=8, min_percentile=80)
#pr.multi_image_plot([diam_dict1['diameter'], diam_dict1['length'], diam_dict1['objects']],
#                    ['diam', 'length', 'objects'], color_map = "spectral")
#diam_dict1["geometry"]

lw_dict = pr.length_width_filter(skel_dict)

geom_sum = pr.summarize_geometry(lw_dict['geometry'], "helloooo")
#geom_sum

binned_medial, binned_geom = pr.bin_by_diameter(lw_dict['length'], lw_dict['diameter'], [2,4,6], "Helloo")
#binned_geom
#pyplot.imshow(binned_medial, cmap = "spectral")

