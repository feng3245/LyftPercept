from PIL import Image
import os
from os import listdir
from os.path import isfile, join
mypath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'
orgpath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraRGB/'
segfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

#f1 = segfiles[0]
#im = Image.open(mypath+f1)

#pixels = im.load()
#pixelsx = [[0 for x in range(im.size[0])] for y in range(im.size[1])]
#for i in range(0, im.size[1]-150):
#    for j in range(im.size[0]):
#        pixelsx[i][j] = 0
#for i in range(im.size[1]-150,im.size[1]):
#    for j in range(im.size[0]):
#        if pixels[j,i] == (10, 0, 0):
#            pixelsx[i][j] = -10

#nocars = 0

for f in segfiles:
  im = Image.open(mypath+f)
  pixels = im.load() # create the pixel map
  ps = 0
  for i in range(im.size[1]): # for every pixel:
    for j in range(im.size[0]):
        if pixels[j,i] == (10, 0, 0):
            ps += 1
#            pixels[j,i] = (0, 0 ,0)
#        else:
#            if not 'zoom' in f:
#                pixels[j,i] = (pixels[j,i][0] + pixelsx[i][j], 0, 0)
#                if pixels[j,i] != (0,0,0):
#                    ps += 1
  if ps == 0:
      os.remove('./Train/CameraSegAug/'+f)
      os.remove('./Train/CameraRGB/'+f)
      print('removed '+f)
#      newim = Image.new("RGB", im.size)
#      newim.putdata(list(im.getdata()))
#      newim.save('./Train/CameraSegAug/'+f)

#for f in segfiles:
  
