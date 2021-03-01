import cv2
import numpy as np
import glob
import os

direc = 'C:\\Users\\Ben\\Desktop\\labelled_frames\\sample_frames\\'

filenames = os.listdir(direc)

img_array = []
for filename in filenames:
    print(filename)
    img = cv2.imread(os.path.join(direc, filename))
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()