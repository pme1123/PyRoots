#! /bin/python3/

"""
Author: @pme1123
Created: Mar 23rd, 2017


Contents:
neighborhod_filter - Filters candidate objects based on pixels near them

Supporting Functions:
*_skeleton_endpoints - finds endpoints of medial axis skeletons
*_find_orientation - determines the orientation of the medial axis at the endpoints
*_extend_skeleton_to_edges - extend skeleton from endpoints along orientation to the edge of an object.
*_local_neighborhoods - identifies areas flanking candidate objects
*_neighborhood_value_distances - calculate distance of pixel values in different edge neighborhoods. 
"""

from scipy import ndimage
import numpy as np
from skimage import img_as_float, measure, morphology

def neighborhood_filter(pyroots_dictionary, 
                        band, 
                        max_dist=0.5, 
                        gap=5, 
                        edge_width=5, 
                        percentiles=(25, 75), 
                        cum_area=67, 
                        return_distance=False):
    """
    Filter objects based on the pixels near them. Looks at the values of 
    pixels on, say, the left side of an object (the "left edge neighborhood")
    and compares them to those in the right edge neighborhood. If the 
    difference is greater than `max_dist`, the object is rejected.
    
    Because this filter works on individual objects in a binary image, it is 
    slow for images with many objects.
    

    Parameters
    ----------
    pyroots_dictionary: dict
        Named objects returned from `pyroots.skeleton_with_distance`. 
        `"objects"` and `"diameter"` are essential.
    band: 2darray
        Grayscale image or band. Dimensions should be the same as `"objects"` 
        in `pyroots_dictionary`
    max_dist: [0, 1]
        Maximum difference between values calculated for each edge 
        neighborhood. Note that `band` is converted to float64. 
    gap: int
        Size of gap between the object and the edge neighborhoods, in pixels. 
        See Notes.
    edge_width: int
        Width of edge neighborhoods, in pixels. This is the region of interest 
        for this function. See Notes
    percentiles: tuple or list [0, 100]
        Lower and upper bounds of percentiles used to calculate the mean value 
        of each edge neighborhood. See Notes.
    cum_area: float [0, 100]
        Only calculate distances between the minimum set of neighborhoods that 
        total this percentage of the total area of neighborhoods. Effectively 
        removes small neighborhoods that might be outliers and falsely eliminate 
        candidate objects later on. 
    return_distance: bool
        Return a list of max distances for each object? 
    
    Returns
    -------
    An updated pyroots_dictionary
    
    *optional* an ndarray of labelled edge neighborhoods from the original 
    objects. 
        
    Notes
    -----
    Images are converted to float64 before analysis. Therefore, distances 
    are bound from [0, 1]. 
    
    The algorithm works as such:
    1. Edge neighborhoods are identified for each object as such: First, the 
    object is dialated by (`gap + edge_width`) pixels. The skeleton is then 
    extended to the edge of the dialated object. Finally, the gap, extended 
    skeleton, and original objects are removed from the fully dialated object. 
    
    2. Each edge neighborhood receives a value, the middle percentiles of pixel 
    values within each edge neighborhood. The middle percentiles are bound by 
    `percentiles`, with the default being the middle 50% (25, 75). 
    
    3. Many objects have multiple edge neighborhoods. In this case, a square 
    distance matrix is calculated between each, where each cell i, j corresponds 
    to the absolute value of the difference between edge neighborhoods i and j. 
    
    4. If any value in the distance matrix is greater than `max_dist`, the object 
    is rejected.
    """
    # housekeeping
    if edge_width < 2:
        edge_width = 2
        print("WARNING: Edge width must be >= 2 for proper functioning. \
            Raising to 2")
    
    band = img_as_float(band)
    
    # working arrays
    img = pyroots_dictionary["objects"]  # binary array of candidate objects
    dist = pyroots_dictionary["diameter"] / 2  # skeleton diameter --> radius
    
    # maximum distance of dilation
    total_dilation = gap + edge_width
    dims = img.shape
    
    # prepare img
    labels, labels_ls = ndimage.label(img)
    props = measure.regionprops(labels)
    
    test = [False] * (labels_ls + 1)
    mx = [0] * (labels_ls + 1)
    
    for i in range(1, labels_ls+1):
        # Bounds of slice to only the object of interest
        # include a gap. Stay within bounds of image.
        a, b, c, d = props[i-1].bbox
        a = max(a - total_dilation, 0)  
        b = max(b - total_dilation, 0)
        c = min(c + total_dilation, dims[1])
        d = min(d + total_dilation, dims[0])
        
        # slice
        temp_img = labels[a:c, b:d] == i
        temp_dist = dist[a:c, b:d] * temp_img
        temp_skel = temp_dist > 0
        temp_band = band[a:c, b:d]
        
        # find neighborhoods
        locs = _local_neighborhoods(temp_img, temp_skel, temp_dist, 
                                    gap, edge_width)
        
        # calculate distances
        dist_mat = _neighborhood_value_distances(locs, temp_band, percentiles, cum_area)
        
        # Does the object meet criteria?
        mx[i] = dist_mat.max()
        test[i] = dist_mat.max() < max_dist
    
    # Update objects
    objects_out = np.array(test)[labels]
    
    # Update geometry
    test[0] = True  # 0 is always kept in geometry dataframe
    geometry_out = pyroots_dictionary["geometry"][test]
    
    out = {"objects"  : objects_out,
           "diameter" : pyroots_dictionary["diameter"] * objects_out,
           "length"   : pyroots_dictionary["length"] * objects_out,
           "geometry" : geometry_out}
    
    if return_distance == True:
        out = [out, mx]
    
    return(out)



def _skeleton_endpoints(skeleton):
    """
    Find the end points of medial axis skeletons
    
    Parameters
    ----------
    skeleton: ndarray
        binary medial axis image
    
    Returns
    -------
    A binary ndarray showing the endpoints of skeletons
    
    """
    
    end_points = ndimage.convolve(1*skeleton, np.ones((3,3))) == 2
    end_points = end_points * skeleton
    return(end_points)
    
    
   
def _find_orientation(skeleton, endpoints):
    """
    Determine the orientation of medial axis skeleton endpoints
    
    Parameters
    ----------
    skeleton: ndarray
        binary medial axis image
    endpoints: ndarray
        binary image identifying endpoints of the medial axis
        
    Returns
    -------
    A list of 3x3 binary arrays that act as structuring elements to extend a
    skeleton along its orientation at each endpoint. Order is determined by
    `scipy.ndimage.label`. 
    
    """
    skeleton = 1*skeleton
    endpoints = 1*endpoints
    
    # custom structure recognizes endpoints of skeletons of length=2 as discrete
    labels, labels_ls = ndimage.label(endpoints, structure=[[0, 0, 0], 
                                                            [0, 1, 0], 
                                                            [0, 0, 0]]) 
    
    out = [np.zeros((3,3))]
    for i in range(1, labels_ls + 1):
        # subset end point
        temp = labels == i
        
        # dilate to kernel size.
        orientation = morphology.binary_dilation(temp, np.ones((3,3)))
        # extract kernel
        orientation = np.ma.masked_array(skeleton, ~orientation).compressed()
        
        # rotate 180 degrees for functionality
        x = len(orientation)-1
        orientation = [orientation[x-j] for j in range(x+1)]
        orientation = np.reshape(orientation, (3, 3))
        
        # add to list
        out.append(orientation)
    return(out)
    


def _extend_skeleton_to_edges(objects,
                              skeleton,
                              endpoints,
                              kernels,
                              lengths):
    """
    Extends `skeleton` from endpoints to edges of `objects` or by `lengths` pixels,
    whichever is shorter.
    
    Parameters
    ----------
    objects: ndarray
        binary image of candidate objects
    skeleton: ndarray
        binary image of medial axis of `objects`
    endpoints: ndarray
        binary image of endpoints of `skeleton`
    kernels: list of ndarrays
        denote orientation of skeleton at each `endpoints`
    lengths: list of int
        denote number of pixels to extend  of objects at each `endpoints`
    
    Returns
    -------
    A binary ndarray of the skeleton, extended.
    
    """

    # custom structure identifies endpoints of objects containing 2 pixels
    labels, labels_ls = ndimage.label(endpoints, structure = [[0, 0, 0], 
                                                              [0, 1, 0], 
                                                              [0, 0, 0]])
    
    out = np.zeros(objects.shape)
    for i in range(1, labels_ls+1):
        # subset
        temp = 1*labels == i
        
        # extend
        times = int(lengths[i-1]) + 2
        for j in range(1, times):
            temp = ndimage.convolve(temp, kernels[i])
            out += temp
    
    out += skeleton
    out = out > 0
    out = out * objects
    
    return(out)
    


def _local_neighborhoods(binary_image, 
                         skeleton, 
                         distance, 
                         gap=0, 
                         edge_width=2):
    """
    Identify neighboring areas of candidate objects, separated to denote each side.
    
    Parameters
    ----------
    binary_image: ndarray
        binary candidate objects image
    skeleton: ndarray
        binary medial axis of objects in `binary_image`
    distance: ndarray
        like `skeleton`, but with each pixel value the radius of the object
    gap: int
        Number of pixels space to leave between neighborhoods and objects
    edge_width: int
        Number of pixels wide to make the neighborhoods
    
    """
    
    if edge_width < 2:
        edge_width = 2
        print("WARNING: Edge width must be >= 2 for proper \
            functioning. Raising to 2")

    
    # find end points
    endpoints = _skeleton_endpoints(skeleton)
    
    # find orientations
    orientations = _find_orientation(skeleton, endpoints)
    
    # find necessary extension lengths
    extension_lengths = np.ma.masked_array(distance, endpoints==0).compressed()
    extension_lengths = extension_lengths + gap + edge_width + 1  
        # for good measure. Shouldn't cause problems. 
    
    # dialate original image to edges (over gap)
    gap = morphology.binary_dilation(binary_image, 
                                     selem=morphology.disk(gap))
    dialated = morphology.binary_dilation(gap, 
                                          selem=morphology.disk(edge_width))
    
    # extend the skeleton
    extended_skel = _extend_skeleton_to_edges(dialated, skeleton, 
                                              endpoints, orientations,
                                              extension_lengths)
    
    
    out = dialated * ~gap * ~extended_skel
    
    return(out)



def _neighborhood_value_distances(local_neighborhoods,
                                  grayscale_image,
                                  percentiles=(25, 75),
                                  cum_area=67):
    """
    Calculate the distance between neighborhoods in terms of pixel values of a
    corresponding grayscale image.
    
    Parameters
    ----------
    local_neighborhoods: binary ndarray
        Edges segmented by _local_neighborhoods, showing pixels along each 
        side of candidate objects
    grayscale_image: ndarray
        Band or grayscale image from which to draw values.
    percentiles: tuple of float [0, 100]
        Lower and upper percentiles to use when calculating the mean. 
        This approach reduces susceptibility to both outliers and bimodal
        distributions with unused values in the middle. 
    cum_area: float [0, 100]
        Only calculate distances between the minimum set of neighborhoods 
        that total this percentage of the total area of neighborhoods. 
        Effectively removes small neighborhoods that might be outliers
        and falsely eliminate candidate objects later on. 
    
    Returns
    -------
    A square distance matrix of differences in 'mean' pixel values between 
    individual neighborhoods in `local_neighborhoods`
    
    """
    
    grayscale_image = img_as_float(grayscale_image)
    
    # Subset the cumulative sum of 66% of total
    # label and sum each area
    labels, labels_ls = ndimage.label(local_neighborhoods)
    total_area = np.sum(local_neighborhoods)
    part_areas = ndimage.sum(local_neighborhoods, labels, index=range(labels_ls+1))
    
    # Accumulate 66% of total in order from largest to smallest areas
    sorted_areas = np.sort(part_areas)[::-1]
    i = j = 0
    while i < total_area*(cum_area/100):
        i += sorted_areas[j]
        j += 1

    new_neighborhoods = [i in sorted_areas[:j] for i in part_areas]
    new_neighborhoods = np.array(new_neighborhoods)[labels] > 0
    
    # Find mean values of each object
    labels, labels_ls = ndimage.label(new_neighborhoods)
    if labels_ls > 1: # need more than one object

        mean_value = []
        for i in range(1, labels_ls + 1):
            temp = labels==i
            # pull initial values
            vals = np.ma.masked_array(grayscale_image, ~temp).compressed()

            # calculate thresholds of these values
            low = np.percentile(vals, percentiles[0])
            high = np.percentile(vals, percentiles[1])

            # subset initial values
            low = vals >= low
            high = vals <= high
            temp = low * high
            avg = np.ma.masked_array(vals, ~temp).compressed()
            mean_value.append(np.mean(avg))

        # distances. Returns a square distance, which shouldn't matter
        dist = np.array([i - mean_value for i in mean_value])
        dist = np.abs(dist)
    
    else:
        dist = np.array(np.NaN)
    
    return(dist)



