

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import sys
import os
from tkinter import *
from tkinter import filedialog

root = Tk()
root.filename = filedialog.askopenfilename(initialdir = '/', title = 'Select the video to annotate')

path = root.filename

direc = os.getcwd()

if not os.path.exists(direc + '\\coordinates'):
    os.mkdir(direc + '\\coordinates')

cap = cv2.VideoCapture(path)
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frameRate = cap.get(cv2.CAP_PROP_FPS)
frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
vid_length =  frameCount/frameRate
buf = np.empty((10, frameHeight, frameWidth, 3), np.dtype('uint8'))
fc = 0
ret = True

while (fc < 1  and ret):
    ret, buf[fc] = cap.read()
    fc += 1
cap.release()
frame = buf[0]


def selectROI(region): 
    
    import pylab
    from matplotlib.widgets import PolygonSelector

    fig, ax = plt.subplots()
    ax.imshow(frame)
    plt.title(region)

    def onselect(verts):
        np.savetxt(direc + '\\coordinates\\' + region, verts)

    polygon = PolygonSelector(ax, onselect)

    plt.show()

locations = ['x_chamber', 'y_chamber', 'left_side', 'right_side', 'middle']

for location in locations:
    selectROI(location)