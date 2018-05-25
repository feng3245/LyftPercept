#!/bin/bash

# Needed for proper apt-get on gpu possibly
# apt-get install apt-transport-https  

# May need to uncomment and update to find current packages
# apt-get update

# Required for demo script! #
git config --global user.email "feng3245@gmail.com"
git config --global user.name "Feng Liu"
pip install scikit-video
pip install opencv-python
pip install Pillow
python preinstall.py
# Add your desired packages for each workspace initialization
#          Add here!          #