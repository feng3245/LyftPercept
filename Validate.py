from PIL import Image
import os
from os import listdir
from os.path import isfile, join
mypath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'

segfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for f in segfiles:
  im = Image.open(mypath+f)
  for d in im.getdata():
    if d != (7, 0, 0) and d != (10, 0, 0) and d != (0, 0, 0):
      print('Bad data')