import cv2
import numpy as np
import tiffile as tiff

images = tiff.imread('C:/DB/ISBI/train-volume.tif')
print(images.shape)

for image in images:
    # print(i.shape)
    cv2.imshow('show', image)
    cv2.waitKey(0)