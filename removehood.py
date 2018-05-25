from PIL import Image
import os
from os import listdir
from os.path import isfile, join
mypath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSeg/'
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

for f in [sf for sf in segfiles if not 'zoom' in sf]:
  im = Image.open(mypath+f)
  orgim = Image.open(orgpath+f)
  im = im.crop((100, 0, 700, 450))
  im = im.resize((800, 600))
  orgim = orgim.crop((100, 0, 700, 450))
  orgim = orgim.resize((800, 600))
  im.save('./Train/CameraSegAug/'+f)
  orgim.save(orgpath+f)


#  pixels = im.load() # create the pixel map
#  ps = 0
#  for i in range(im.size[1]): # for every pixel:
#    for j in range(im.size[0]):
#      if pixels[j,i] == (6, 0, 0) or pixels[j,i] == (7, 0, 0):
#        pixels[j,i] = (7, 0 ,0)
#      else:
#        if pixels[j,i] != (10, 0, 0):
#            pixels[j,i] = (0, 0 ,0)
#        else:
#            if not 'zoom' in f:
#                pixels[j,i] = (pixels[j,i][0] + pixelsx[i][j], 0, 0)
#                if pixels[j,i] != (0,0,0):
#                    ps += 1
#  if not 'zoom' in f:
#      newim = Image.new("RGB", im.size)
#      newim.putdata(list(im.getdata()))
#      newim.save('./Train/CameraSegAug/'+f)

#for f in segfiles:
  
