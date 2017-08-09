: <<DOC

Author: pme1123
Created Aug 8th, 2017
www.github.com/pme1123/pyroots


Installation Instructions: Via command line for ubuntu-linux systems (tested on Mint 17).

Can also run this as a shell script. If you do this, and want to skip openCV, then comment out everything starting with Step 5. 

If you encounter an error, it's probably from installing python modules. Check the dependencies for each module on stackexchange or similar. Alternatively, download and install the Ananconda python3 distribution, then install pip (sudo apt-get install python3-pip)

This process will take a while...

1. Install python3, pip, and dependencies. Git is optional, but convenient for downloading pyroots and opencv from github. The others are required dependencies for numpy, matplotlib, and scikit-image.
DOC

sudo apt-get install python3 python3-dev python3-pip ipython3 ipython3-notebook 

sudo apt-get install libatlas-base-dev gcc gfortran g++ libopenblas-dev liblapack-dev libfreepack6-dev libpng12-dev libjpeg-dev git


: <<DOC
2. Install python modules to user space (or use sudo and remove the --user flag to install for everyone)
DOC

pip3 install --user numpy scipy scikit-image matplotlib pandas os multiprocessing colour-science time tqdm



: <<DOC
3. Install jupyter notebooks
DOC

pip3 install --user jupyter
echo 'export PATH=~/.local/bin:$PATH' >>$HOME/.bash_profile  # Append the path to jupyter to your bash profile.
source ~/.bash_profile

: <<DOC
4. Download and install pyroots
DOC

cd $HOME/.local/lib/python3.*
git clone https://github.com/pme1123/pyroots.git
mkdir $HOME/pyroots

# make symbolic links of interactive documents for easy updating using `git pull`
ln -s "pyroots/Example Images" "$HOME/pyroots/Example Images"
ln -s "$HOME/.local/lib/python3.*/pyroots/Notebooks for Parameterization" "$HOME/pyroots/Notesbooks for Parameterization"
ln -s "pyroots/Command Line Scripts" "$HOME/pyroots/Command Line Scripts"
ln -s "pyroots/README.md" "$HOME/pyroots/README.md"
ln -s "pyroots/Update_Pyroots.sh" "$HOME/pyroots/Update_Pyroots.sh"

cd $HOME

: <<DOC
4. Install dependencies for openCV (for preprocessing functions, optional for thresholding functions)
DOC

sudo apt-get install build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

# optional. Make operations faster. 
sudo apt-get install libeigen3-dev libtbb-dev  



: <<DOC
5. Clone the latest version of OpenCV to the file of your choice
DOC

mkdir $HOME/Programs # or wherever you want to put openCV
cd $HOME/Programs
git clone https://github.com/opencv/opencv.git   # clones it to <path>/opencv



: <<DOC
6. Install OpenCV
DOC
    
cd opencv
mkdir build
cd build
    
# prep for installation
#cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..

# or if using eigen and/or tbb for faster operations:
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_EIGEN=ON -D WITH_TBB=ON ..
    
# compile for installation
make
    
# install
sudo make install   
