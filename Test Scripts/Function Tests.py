# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 10:47:44 2016

@author: patrick

Test core functions in pyroots. Return a 
"""

import pyroots as pr
from skimage import io
from matplotlib import pyplot

importlib.reload(pr)

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
length_raw, dist_raw, geom_raw = pr.skeleton_with_distance(noise)
#pr.multi_image_plot([length_raw, dist_raw], ['length', 'diam'], 
#                    color_map = "spectral")

diamfilt, dist_diamfilt, length_diamfilt, geom_diamfilt = pr.diameter_filter(dist_raw, length_raw, noise, max_diameter=8, max_percentile=80)
#pr.multi_image_plot([diamfilt, dist_diamfilt, length_diamfilt], 
#                    ['obj','diameter','length'], 
#                    color_map = "spectral")
#geom_diamfilt

diamfilt1, dist_diamfilt1, length_diamfilt1, geom_diamfilt1 = pr.diameter_filter(dist_raw, length_raw, noise, min_diameter=8, min_percentile=80)
#pr.multi_image_plot([diamfilt1, dist_diamfilt1, length_diamfilt1], 
#                    ['obj','diameter','length'], 
#                    color_map = "spectral")

lwf, geom_lwf = pr.length_width_filter(diamfilt, geom_diamfilt)

geom_sum = pr.summarize_geometry(geom_lwf, "helloooo")
#geom_sum

binned_medial, binned_geom = pr.bin_by_diameter(length_diamfilt, dist_diamfilt, [2,4,6], "Helloo")
#binned_geom
#pyplot.imshow(binned_medial, cmap = "spectral")

