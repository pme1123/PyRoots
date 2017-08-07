#!/bin/python3

# Interactive script to run a segmentation and medial axis loop using pyroots. 
# Steps:
#   1. Copy this script to the directory you want to work in.
#       - images and settings should be in a child directory!
#   2. Open your terminal and navigate to the working directory (ex. with `cd "path_to_directory"`
#   3. type: `python3 script_name.py`
# If you'll run this a bunch of times, try editing the non-interactive version

#TODO: DEBUG

import pyroots as pr
from os import path, getcwd
import warnings

# Parameters
## method
method = input("What is your analysis method? ['frangi', 'thresholding']").lower()
while method != 'frangi' and method != 'thresholding':
    method = input("Try again: ['frangi', 'thresholding']")
    

## input directory
root_dir = getcwd()

dir_in = input("What is the target directory in {}? ".format(root_dir))

dir_in = path.join(root_dir, dir_in)
while not path.exists(dir_in):
    dir_in = input("Path {} doesn't exist!\nWhat is the target directory in {}? "\
        .format(dir_in, root_dir))
    dir_in = path.join(root_dir, dir_in)


## output images directory
dir_out_t = path.join(root_dir, "Pyroots Out")

dir_out = input("Writing output to:\n\t{}.\n\n\tIs this OK? [y, n]"\
                .format(dir_out_t)).upper()
while dir_out != "Y" and dir_out != "N":
    dir_out = input("Try again: [y, n]").upper()

if dir_out == "Y":
    dir_out = dir_out_t
else:
    dir_out = input("Where would you like to write output images\n(in {})?".format(root_dir))
    dir_out = path.join(root_dir, dir_out)

## Output table
tab_out_t = path.join(root_dir, "output {}.txt".format(123))####DATE))
tab_out = input("Write data to:\n\n\t{}\n[y, n]".format(tab_out_t)).upper()
while tab_out != "Y" and tab_out != "N":
    tab_out = input("Try again: [y, n]")

if tab_out == "Y":
    tab_out = tab_out_t
else:
    tab_out = input("What would you like to call the output table?")

## Table overwrite
q = "N"
while q == "N":
    if path.exists(path.join(root_dir, tab_out)):
        tab_over = input("Table exists. Append? [y, n]").upper()
        while tab_over != "Y" and tab_over != "N":
            tab_over = input("Try again: [y, n]").upper()

        tab_over = tab_over == "Y"
        
        if tab_over == False:
            q = input("Overwrite existing data? [y, n]").upper()
            while q != "Y" and q != "N":
                q = input("Try again: [y, n]").upper()


## Parameters
params = input("What is the name of the parameters file?")
params = path.join(root_dir, params)
while not path.exists(params):
    params = input("Couldn't find {}.\n\nWhat is the name of the parameters file?".format(params))
    params = path.join(root_dir, params)


## extensions
extension_in = input("What is the image type for input images?")
extension_out = ".png"

print("Reading in {}. Writing out {}. To change this, edit the script".format(extension_in, extension_out))


## Multiprocessing threads
threads = int(input("How many cores would you like to use?"))


## Begin loop. 
warnings.filterwarnings("ignore")

x = pr.pyroots_batch_loop(dir_in,
                          dir_out=dir_out,
                          extension_in=extension_in, 
                          method=method,
                          table_out=tab_out,
                          params=params,
                          save_images=True,
                          table_overwrite=tab_over,
                          threads=threads)
