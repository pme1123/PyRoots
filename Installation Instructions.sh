#! /bin/bash

: <<DOC

Author: pme1123
Created Aug 8th, 2017
Updated Nov 2nd, 2018
www.github.com/pme1123/pyroots


Installation Instructions: Via command line for ubuntu-linux systems (tested on Mint 17). Can also run this as a shell script.

If you encounter an error, it's probably from installing python modules. Check the dependencies for each module on stackexchange or similar. Alternatively, download and install the Ananconda python3 distribution, then install pip (sudo apt-get install python3-pip).

This process will take a while...

1. Install python3, pip, and dependencies. Git is optional, but convenient for downloading pyroots from github. The others are required for numpy, matplotlib, and scikit-image.
DOC

sudo apt-get install python3 python3-dev python3-pip ipython3

sudo apt-get install libatlas-base-dev gcc gfortran g++ libopenblas-dev liblapack-dev libpng-dev libjpeg-dev git


: <<DOC
2. Install python modules to user space (or use sudo and remove the --user flag to install for everyone)
DOC

#MS=( numpy scipy scikit-image matplotlib pandas colour-science tqdm cv2 )

#for i in ${MS[@]}; do
#  pip3 install --user $i
#done

pip3 install --user numpy scipy scikit-image matplotlib pandas colour-science tqdm cv2


: <<DOC
3. Install jupyter notebooks
DOC

pip3 install --user jupyter
echo 'export PATH=~/.local/bin:$PATH' >>$HOME/.bash_profile  # Append the path to jupyter to your bash profile.
source ~/.bash_profile

: <<DOC
4. Download and install pyroots. This method clones it from github into your ~/.local/lib/python3 folder, then makes symbolic links from the interactive documents in a new folder called ~/pyroots. 
DOC

# download pyroots from github
cd $HOME/.local/lib/python3.*
cd site-packages
git clone https://github.com/pme1123/pyroots.git

# install pyroots
python3 pyroots/setup.py install --user

# make symbolic links of interactive documents for easy updating using `git pull`
mkdir $HOME/pyroots
ln -s "pyroots/Example Images" "$HOME/pyroots/Example Images"
ln -s "pyroots/Notebooks for Parameterization" "$HOME/pyroots/Notesbooks for Parameterization"
ln -s "pyroots/Command Line Scripts" "$HOME/pyroots/Command Line Scripts"
ln -s "pyroots/README.md" "$HOME/pyroots/README.md"
ln -s "pyroots/Update_Pyroots.sh" "$HOME/pyroots/Update_Pyroots.sh"

cd $HOME 
