
# Pyroots: A Python module to quantify roots and hyphae in low-quality, noisy images.

1. Requirements
1. Overview
2. Installation
3. Workflow
4. Segmentation modes: Thresholding (see notebook)
5. Segmentation modes: Vessel enhancement (see notebook)
6. Preprocessing options (see notebooks)
7. Hyphae Extraction (see protocol)
7. Functions (see source code)

## Requirements
- Internet (for installing)
- Up to 2 GB RAM per CPU core you'll use (this varies with image size)
- Some coding experience, though not necessarily in Python. If you know *R*, you're fine. There's lots of example code here.

The installation script should work with most linux distributions that use `APT` (e.g. Ubuntu, Mint, Debian).
You should be able to follow it for mac or windows.

For photomicroscopy, you'll need:

- A brightfield (aka regular) microscope with a 10x objective. The objective should be planar quality or better to minimize [chromatic abberation](https://digital-photography-school.com/chromatic-aberration-what-is-it-and-how-to-avoid-it/).
- A camera that can produce images of at least 300 PPI. Smartphones work in a pinch if you have an adapter and a camera app that gives you a lot of manual control. I used an older [spot](http://www.spotimaging.com/cameras/).
- Lots of storage space

The included 'rapid' hyphae extraction method uses aniline blue, which doesn't require fluorescence.

## Overview

Hyphae and root distribution are hugely important for driving belowground processes, but also a huge pain to quantify.
This is unfortunate, because distributions are also highly variable so you need a lot of samples to get good estimates.
Image analysis is an upgrade over manual measurements because it's much! faster than manually measuring, AND it isn't
prone to user bias. This means you can have anyone take images, and as long as they follow protocol, you'll get the
same result. That said, **the number you get out is quite sensitive to how you parameterize the program**, which may or may
also affect precision, rank-order, and your ability to detect differences.

For roots, lots of options already exist. [WinRhizo](http://regent.qc.ca/assets/winrhizo_about.html) is an excellent program
that uses thresholding to separate roots from dirt and background. It can do some things pyroots doesn't, like measure
topology and estimate volume. (if you want to contribute this ability, let me know!). It has some challenges with
environmental samples (see below). [ImageJ](https://imagej.nih.gov/ij/) is another option. I have limited experience
with this program.

The basic workflow is to identify pixels in an image that are likely to belong to objects (roots or hyphae), and not
background, dirt, or plant residue. These are turned into a binary image, in which groups of `TRUE` pixels form objects.
Then, a medial axis skeletonization algorithm finds the center line(s) of each object, or "topological skeleton", and
the length of that skeleton is the object length (more or less). For a good overview, see the
[tutorial](http://scikit-image.org/docs/0.10.x/auto_examples/plot_medial_transform.html) at
[scikit-image](http://scikit-image.org/). These objects can also be filtered based on their geometry
(length:width ratio, mean diameter, etc).

WinRhizo is great for the greenhouse samles, but in my experience, it doesn't do very well with environmental samples
where the roots are a variety of colors. The color differences create speckles in the thresholded image, which splits
the skeleton and so inflates the object length. In my testing, this ends up being a 100% inflation vs the number you
get with `pyroots`. This is fine if you just want to find differences among treatments, because it's consistent. Your
results will vary. Measuring hyphal length density is near-impossible with WinRhizo because the hyphae are so darn faint sometimes, and the microscope introduces a lot of visual artifacts and isn't always in perfect focus.

Pyroots gets over these issues in a couple of ways:

- Binary morphological functions to remove speckles
- Vessel enhancement to emphasize faint objects. 
- A much richer set of object filtering functions than WinRhizo, such as solidity and diameter
- The ability to use most any colorspace to classify pixels
- Multiprocessing, so you can run multiple images at once in batch mode, limited by the size of your cluster.
- Functions to improve the quality of microscope images.

## Installation
##### Run the installation script
Follow the instructions in the [`Installation Instructions.sh`](https://github.com/pme1123/pyroots/blob/master/Installation%20Instructions.sh) script. Note this was tested on Linux Mint 17.1. You may be able to execute it, but will need admin rights. Download the script, then try:

```bash
cd <path_to_script>
chmod +x "Installation Instructions.sh"
sudo sh "Installation Instructions.sh"
```

This does the following:

1. Installs python3: python3 python3-dev python3-pip ipython3 ipython3-notebook  **requires root**
2. Installs dependencies: libatlas-base-dev gcc gfortran g++ libopenblas-dev liblapack-dev libfreepack6-dev libpng12-dev libjpeg-dev git **requires root**
3. Installs supporting python3 modules to your userspace: numpy scipy scikit-image matplotlib pandas os multiprocessing colour-science time tqdm
4. Makes a directory for pyroots in your local .python3.* folder, and clones the pyroots repository there.
5. Makes a directory in your home folder for interactive pyroots documents, and puts in some symbolic links. **These will upate as you run the update script, so save your work in a different file**
6. **Optional**: Install OpenCV. This takes a while, so only do it if you're using the image preprocessing functions (i.e. for photomicroscopy).

## Workflow

1. Acquire your images. See the suggested protocol for photomicroscopy. The images should be saved with a filename that consistently identifies the sample and the image number for the sample, e.g. `SampleID_0001.tif`.

2. Select a subset for setting parameters. I suggest up to 30 random images.

2. Set parameters for image quality improvement ("preprocessing"):

    1. Filtering (removes blurry images and otherwise faulty images). Found in [preprocessing_filtering.ipynb](link)
    2. Smooths, color-balances, and removes chromatic abberation. Found in [preprocessing_actions.ipynb](link)

3. Export the parameters and preprocess in batch mode. I usually run the filtering script as I move images from the microscope computer to long-term storage, as it compresses the files (losslessly) automatically and reduces the number of images to store.

3. Set parameters for thresholding. For hyphae images, I recommend the [Frangi_segmentation.ipynb](link) approach, which uses a vessel enhancement step. For roots, the [thresholding_segmentation.ipynb](link) approach should be adequate and is easier to parameterize.

5. Test and export the parameters.

6. Run in batch mode on a larger subset of your images. 

7. Inspect results. Tweak parameters as necessary. Repeat.

8. Run on all images.

7. Analyze the output using the tools of your choice.

## Output
Pyroots will return the following:

- A binary image showing the objects measured
- A table summarizing the length and/or diameter of objects in each image. Units are PIXELS.
