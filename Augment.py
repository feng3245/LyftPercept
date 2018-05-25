from PIL import Image
import os
from os import listdir
from os.path import isfile, join
mypath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'
orgpath = os.path.dirname(os.path.realpath(__file__))+'/Train/CameraRGB/'
segfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

f1 = segfiles[0]
im = Image.open(mypath+f1)
pixels = im.load()
pixelsx = [[0 for x in range(im.size[0])] for y in range(im.size[1])]
for i in range(0, im.size[1]-150):
    for j in range(im.size[0]):
        pixelsx[i][j] = 0

for i in range(im.size[1]-150,im.size[1]):
    for j in range(im.size[0]):
        if pixels[j,i] == (10, 0, 0):
            pixelsx[i][j] = -10

nocars = 0

for f in segfiles:
  im = Image.open(mypath+f)
  pixels = im.load() # create the pixel map
  ps = 0
  for i in range(im.size[1]): # for every pixel:
    for j in range(im.size[0]):
      if pixels[j,i] == (6, 0, 0) or pixels[j,i] == (7, 0, 0):
        pixels[j,i] = (7, 0 ,0)
      else:
        if pixels[j,i] != (10, 0, 0):
            pixels[j,i] = (0, 0 ,0)
        else:
            if not 'zoom' in f:
                pixels[j,i] = (pixels[j,i][0] + pixelsx[i][j], 0, 0)
                if pixels[j,i] != (0,0,0):
                    ps += 1
  if ps == 0:
    nocars += 1
  if not os.path.isfile(os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'+f): 
    newim = Image.new("RGB", im.size)
    newim.putdata(list(im.getdata()))
    newim.save(os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/'+f)
  if not os.path.isfile(os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/zoom'+f):
    imorg = Image.open(orgpath+f)
    imgcropped = imorg.crop((200, 150, 600, 450))
    imgcropped = imgcropped.resize((800,600))
    imgcropped.save(orgpath+'zoom'+f)
    imaug = im.copy()
    imaugcropped = imaug.crop((200, 150, 600, 450))
    imaugcropped = imaugcropped.resize((800,600))
    imaugcropped.save(os.path.dirname(os.path.realpath(__file__))+'/Train/CameraSegAug/zoom'+f)
print('no cars {0}.'.format(nocars))   

#for f in segfiles:
  
