from PIL import Image
import os
from os import listdir
from os.path import isfile, join
mypath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'
segfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
carpix = 0
roadpix = 0
otherpix = 0
for f in segfiles:
  im = Image.open(mypath+f)
  pixels = im.load() # create the pixel map
  for i in range(im.size[0]): # for every pixel:
    for j in range(im.size[1]):
      if pixels[i,j] == (7, 0, 0):
        roadpix += 1
      else:
        if pixels[i,j] == (10, 0, 0):
            carpix += 1
        else:
            otherpix += 1
            
print('cars {0}.'.format(carpix/(otherpix+carpix+roadpix)))   
print('roads {0}.'.format(roadpix/(otherpix+carpix+roadpix)))
#for f in segfiles:
  
