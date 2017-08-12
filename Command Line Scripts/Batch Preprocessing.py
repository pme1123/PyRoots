#!/bin/python3

# Interactive script to run a segmentation and medial axis loop using pyroots. 
# Steps:
#   1. Copy this script to the directory you want to work in.
#       - images and settings should be in a child directory!
#   2. Open your terminal and navigate to the working directory (ex. with `cd "path_to_directory"`
#   3. type: `python3 script_name.py`
# If you'll run this a bunch of times, try editing the non-interactive version

import pyroots as pr
from os import path, getcwd, mkdir
import warnings


## Base directory
root_dir = getcwd()

## input/output directories
dir_in = 'Test Images/preprocessing test images'
dir_in = path.join(root_dir, dir_in)

dir_out = "Test Images/preprocessing test out"
dir_out = path.join(root_dir, dir_out)

## Extensions
extension_in = '.png'
extension_out = '.png'  # recommended

## Parameters path
params = "Settings/preprocessing_actions_parameters_08-11-2017.py"
params = path.join(root_dir, params)

## Multiprocessing
threads = 3


##########################################################################
##########################################################################
##########################################################################
##########################################################################
        
print("Directories and analysis parameters:\n\n")
p = ['dir_in', 'extension_in', 'dir_out', 'extension_out', 'params', 'threads']
for i in p:
    print("-- {} = {}".format(i, str(locals()[i])))
    
q = input("Do these parameters look good? [y, n] ").lower()
if q == 'n':
    print("Exiting. Edit this script!")
    exit()
else:
    print("Beginning analysis....\n\n\n\n")

## Make output directory
if not path.exists(dir_out):
    # identify all folders to create
    pathnames = []
    temp_path = dir_out

    while len(root_dir) < len(temp_path):
        t = path.split(temp_path)
        pathnames.insert(0, t[1])
        temp_path = t[0]
    
    j = 0
    for i in pathnames:
        temp_path = i
        if j > 0:
            write_path = path.join(write_path, i)
        else:
            write_path = temp_path
        try:
            mkdir(path.join(root_dir, write_path))
        except:
            pass
        j += 1
    

## Begin loop. 
warnings.filterwarnings("ignore")

pr.preprocessing_actions_loop(dir_in,
                              extension_in,
                              dir_out,
                              extension_out,
                              params,
                              threads)

