#!/bin/python3

# Interactive script to run a segmentation and medial axis loop using pyroots. 
# Steps:
#   1. Copy this script to the directory you want to work in.
#       - images and settings should be in a child directory!
#   2. Open your terminal and navigate to the working directory (ex. with `cd "path_to_directory"`
#   3. type: `python3 script_name.py`
# If you'll run this a bunch of times, try editing the non-interactive version

import pyroots as pr
from os import path, getcwd
import warnings
import pandas as pd


## Base directory
root_dir = getcwd()

## input/output directories
dir_in = 'images_in'
dir_in = path.join(root_dir, dir_in)

dir_out = "Pyroots Out"
dir_out = path.join(root_dir, dir_out)

## Extensions
extension_in = '.TIF'
extension_out = '.png'  # recommended

## method
method = 'thresholding'

## Parameters path
params = "Parameters_filename.txt"
params = path.join(root_dir, params)

## Output table
tab_out = path.join(root_dir, "output {}.txt".format(123))#DATE))
table_overwrite = False

## Multiprocessing
threads = 4


##########################################################################
##########################################################################
##########################################################################
##########################################################################


## warn if overwriting a data table
if path.exists(tab_out) and table_overwrite == True:
    q = input("""Are you sure you want to overwrite the existing table, in\n{}\n
    [y, n]?""".format(tab_out)).lower()
    if q == 'n':
        print("Exiting. Edit this script!")
        exit()
        
print("Directories and analysis parameters:\n\n")
p = ['dir_in', 'extension_in', 'dir_out', 'extension_out', 'method', 'tab_out', 'params', 'threads']
for i in p:
    print("-- {} = {}".format(i, str(locals()[i])))
    
q = input("Do these parameters look good? [y, n] ").lower()
if q == 'n':
    print("Exiting. Edit this script!")
    exit()
else:
    print("Beginning analysis....\n\n\n\n")

## Begin loop. 
warnings.filterwarnings("ignore")

pr.pyroots_batch_loop(dir_in,
                      extension_in=extension_in, 
                      method=method,
                      dir_out=dir_out,
                      extension_out=extension_out,
                      table_out=tab_out,
                      table_overwrite=table_overwrite,
                      params=params,
                      save_images=True,
                      threads=threads)
