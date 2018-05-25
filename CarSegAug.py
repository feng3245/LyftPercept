from PIL import Image
import os
from os import listdir
from os.path import isfile, join
mypath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'
orgpath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraRGB/'
segfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


for f in segfiles:
  im = Image.open(mypath+f)
  pixels = im.load() # create the pixel map
  ps = 0
  minx = im.size[0]
  miny = im.size[1]
  maxx = 0
  maxy = 0
  for i in range(im.size[1]): # for every pixel:
    for j in range(im.size[0]):
      if pixels[j,i] == (6, 0, 0) or pixels[j,i] == (7, 0, 0):
        pixels[j,i] = (7, 0 ,0)
      else:
        if pixels[j,i] != (10, 0, 0):
            pixels[j,i] = (0, 0 ,0)
        else:
            if pixels[j,i] != (0,0,0):
                ps += 1
                if j < minx:
                    minx = j
                if i < miny:
                    miny = i
                if j > maxx:
                    maxx = j
                if i > maxy:
                    maxy = i
  if float(ps)/(im.size[0]*im.size[1])>0.1:
    ysize = im.size[1]
    xsize = im.size[0]
    newim = im.transpose(Image.FLIP_LEFT_RIGHT).crop((minx, miny, maxx, maxy)).resize((xsize, ysize), Image.BICUBIC)
    newim.save(os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/Car'+f)
    imorg = Image.open(orgpath+f).transpose(Image.FLIP_LEFT_RIGHT).crop((minx, miny, maxx, maxy)).resize((xsize, ysize), Image.BICUBIC)
    imorg.save(orgpath+'Car'+f)

#for f in segfiles:
  
