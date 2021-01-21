
### importing all the necessary packages for the functions to work ###

import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
from shapely.geometry import Point
from shapely.geometry import Polygon
import sys
import os
from tkinter import *
from tkinter import filedialog
import tkinter.font as tkFont
from tkinter import ttk
from PIL import ImageTk, Image

### the first step is to just grab an example video frame so that the user can annotate where the chambers were located for this testing day ###

def grab_video_frame(v_location):

    ## using the cv2 library to open up a video from the analysis directory and create a single frame ##

    cap = cv2.VideoCapture(os.path.join(v_location, os.listdir(v_location)[0]))

    #cap = cv2.VideoCapture(v_location + '\\' + os.listdir(v_location)[0])
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = cap.get(cv2.CAP_PROP_FPS)
    frameCount = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_length =  frameCount/frameRate
    buf = np.empty((10, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    ## there is likely an easier way to do this but this is what I've found and it works ##

    while (fc < 1  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    frame = buf[0]

    ## returns just a single frame, which is a numpy array, that can be annotated in the next step ##

    return frame


def selectROI(region, frame, main_dir): 

    ## function that allows us to display a matplotlib interactive polygon selector to annotate the frame for each region (which is one of the arguments) ##

    fig, ax = plt.subplots()
    ax.imshow(frame)
    plt.title(region)

    ## this is the function that is fed to the polygon selector tool. All I am doing here is saving the coordinates for every polygon drawn to a text file of numpy coordinates ##
    ## this allows me to save them outside of the python kernel, and they can be pulled in later at any point ##
    def onselect(verts):
        np.savetxt(main_dir + '\\coordinates\\' + region, verts)

    polygon = PolygonSelector(ax, onselect)

    plt.show()


def plot_coordinates_frame(frame, coordinates):

        ## plotting the coordinates of the annotation on top of the frame for the user to see ##
        ## this gives the user the chance to change things if they don't look quite right before we calculate the interaction times ##
        fig, ax = plt.subplots(1,1,figsize=(5,3))

        ## once the figure is initialized, I'm building the matplotlib plot 1 annotation at a time with different colors ##
        ## these annotations are brought in from the numpy arrays, are combined into this 'coordinates' variable, which is a list of lists of the coordinates ##
        ## it is made up of 'possible places', which are the 5 zones annotated by the user, and 'extra coords' which are additional zones/points that are calculated and needed for the analysis ##
        ax.plot(coordinates[0]['left_side'][:,0], coordinates[0]['left_side'][:,1], color = 'red', linewidth = 2)
        ax.plot(coordinates[0]['right_side'][:,0], coordinates[0]['right_side'][:,1], color = 'purple', linewidth = 2)
        ax.plot(coordinates[1]['x_chamber'][:,0], coordinates[1]['x_chamber'][:,1], color = 'blue', linewidth = 2)
        ax.plot(coordinates[1]['y_chamber'][:,0], coordinates[1]['y_chamber'][:,1], color = 'green', linewidth = 2)
        ax.plot(coordinates[1]['x_outer'][0], coordinates[1]['x_outer'][1], color = 'blue', linewidth = 2)
        ax.plot(coordinates[1]['y_outer'][0], coordinates[1]['y_outer'][1], color = 'green', linewidth = 2)
        ax.plot(coordinates[1]['x_center'][0], coordinates[1]['x_center'][1], 'bo')
        ax.plot(coordinates[1]['y_center'][0], coordinates[1]['y_center'][1], 'go')
        ax.imshow(frame)
        plt.xlim(0,1280)
        plt.ylim(0,720)
        plt.title('Example Coordinate Overlay')
        plt.show()


def plot_heatmap(coordinates, df, trial_frames):
        fig, ax = plt.subplots(1,1,figsize=(5,3))
        ax.plot(coordinates[0]['left_side'][:,0], coordinates[0]['left_side'][:,1], color = 'red', linewidth = 2)
        ax.plot(coordinates[0]['right_side'][:,0], coordinates[0]['right_side'][:,1], color = 'purple', linewidth = 2)
        ax.plot(coordinates[1]['x_chamber'][:,0], coordinates[1]['x_chamber'][:,1], color = 'blue', linewidth = 2)
        ax.plot(coordinates[1]['y_chamber'][:,0], coordinates[1]['y_chamber'][:,1], color = 'green', linewidth = 2)
        ax.plot(coordinates[1]['x_outer'][0], coordinates[1]['x_outer'][1], color = 'blue', linewidth = 2)
        ax.plot(coordinates[1]['y_outer'][0], coordinates[1]['y_outer'][1], color = 'green', linewidth = 2)
        ax.plot(coordinates[1]['x_center'][0], coordinates[1]['x_center'][1], 'bo')
        ax.plot(coordinates[1]['y_center'][0], coordinates[1]['y_center'][1], 'go')
        ax.plot(df['nose']['x'].loc[trial_frames[0]:trial_frames[1]], df['nose']['y'].loc[trial_frames[0]:trial_frames[1]], c='k', alpha=.3, marker='o', linestyle='None')



def time_df(df_times, v_location):

    ## import the time dataframeand calculate when the sociability and social novelty occur based on start time ##
    ## in the future, it would be nice to be able to specify ONLY soc, or ONLY snp, or both ##
    df_times.dropna(inplace = True)

    df_times['StopSocialSec'] = df_times['StartSocialSec'] + 300
    df_times['StopNovelSec'] = df_times['StartNovelSec'] + 300

    ### we want to iterate through the rows, grab the video name, open corresponding video, extract framerate, and then multiply startSec cols

    frameRate = np.zeros(len(df_times))
    StartSocialFrames = np.zeros(len(df_times))
    StartNovelFrames = np.zeros(len(df_times))
    StopSocialFrames = np.zeros(len(df_times))
    StopNovelFrames = np.zeros(len(df_times))

    for i in range(len(df_times)):

        cap = cv2.VideoCapture(v_location + '\\' + df_times['VideoName'][i] + '.mp4')
        frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
        StartSocialFrames[i] = df_times['StartSocialSec'][i] * frameRate[i]
        StartNovelFrames[i] = df_times['StartNovelSec'][i] * frameRate[i]
        StopSocialFrames[i] = df_times['StopSocialSec'][i] * frameRate[i]
        StopNovelFrames[i] = df_times['StopNovelSec'][i] * frameRate[i]

    df_times['StartSocialFrames'] = StartSocialFrames
    df_times['StartNovelFrames'] = StartNovelFrames
    df_times['StopSocialFrames'] = StopSocialFrames
    df_times['StopNovelFrames'] = StopNovelFrames

    ## returning the prepared df_times, which has the calculated start and stop frames based on the framerate of each video ##
    return df_times



##### Functions for main calculations #####

def check_coords(coords, possible_places):

    ## simple but important function for check which of the 5 annotated places the the given coordinates fall within ##

    x = []

    for i in range(len(list(possible_places.values()))):   
        pt = Point(coords)
        if isinstance(list(possible_places.values())[i], Polygon):
            polygon = list(possible_places.values())[i]
        else:
            polygon = Polygon(list(map(tuple, list(possible_places.values())[i])))
        x = np.append(x, polygon.contains(pt))

    ## returns a list of lists (x) that is 5 x 5 ##
    ## that is, 5 bodypart coordinates (nose/l ear/r ear/tail base/tail end) and 5 possible locations ##
    ## example [0, 1, 0, 0, 1] in position 1 would indicate that the nose body part is in annotated location 2 and 4 (where there is a 1 that body part is in that location) 
    ## the order is the order of possible_places: {'x_zone', 'y_zone', 'left_side', 'middle', 'right_side'} ##
    ## in the example above, the nose would be in the y_zone on the right side ##
    return x 


def check_climbing(df, coords):

    ## simple, alpha function for filtering frames where the animal seems to be climbing, rather than investigating ##

    state = ['climbing'] * len(df)
    z = -1

    ## to test this, I have implemented a simple algorithm that checks the distance between the nose and the ears ##
    ## look ma, I'm using pythagorean theorem! to calculate the distance when i create a triangle between these three coordinates ##
    
    for index, val in df.iterrows():
        z = z + 1
        distance_1 = np.sqrt(((df['left ear']['x'].loc[index] - df['nose']['x'].loc[index])**2) + ((df['left ear']['y'].loc[index] - df['nose']['y'].loc[index])**2))
        distance_2 = np.sqrt(((df['right ear']['x'].loc[index] - df['left ear']['x'].loc[index])**2) + ((df['right ear']['y'].loc[index] - df['left ear']['y'].loc[index])**2))
        distance_3 = np.sqrt(((df['nose']['x'].loc[index] - df['right ear']['x'].loc[index])**2) + ((df['nose']['y'].loc[index] - df['right ear']['y'].loc[index])**2))
    
    ## if the sum of the distances is lower than some arbitrary value, the state is still climbing ##
    ## if the distances are greater (indicating the mouse's head is not visually occluded) than some value, we say that it is 'not climbing' ##
    ## it would be nice in the future to try to figure out a better way to automatically detect climbing behavior with the coordinate outputs ##

        if np.sum(distance_1, distance_2, distance_3) > 5:
            state[z] = 'not_climbing'

def check_orientation_single(df, index_loc, extra_coords):

    ## main function for testing orientation of the head ##
    ## this is also a relatively new addition, based on some coordinate math ##
    ## essentially we are calculating whether nose is closest to the center point of the subject chamber ##
    ## to do this, we are calculating the distances to all three head body parts ##

    orientation = 'not_oriented'
    x_center = extra_coords['x_center']
    y_center = extra_coords['y_center']

    dist_to_x = np.sqrt(((x_center[0] - df['nose']['x'].loc[index_loc])**2) + ((x_center[1] - df['nose']['y'].loc[index_loc])**2))
    dist_to_y = np.sqrt(((y_center[0] - df['nose']['x'].loc[index_loc])**2) + ((y_center[1] - df['nose']['y'].loc[index_loc])**2))
    
    if dist_to_x > dist_to_y:        
        distance_to_nose = np.sqrt(((y_center[0] - df['nose']['x'].loc[index_loc])**2) + ((y_center[1] - df['nose']['y'].loc[index_loc])**2))
        distance_to_l_ear = np.sqrt(((y_center[0] - df['left ear']['x'].loc[index_loc])**2) + ((y_center[1] - df['left ear']['y'].loc[index_loc])**2))
        distance_to_r_ear = np.sqrt(((y_center[0] - df['right ear']['x'].loc[index_loc])**2) + ((y_center[1] - df['right ear']['y'].loc[index_loc])**2))
    elif dist_to_x < dist_to_y:
        distance_to_nose = np.sqrt(((x_center[0] - df['nose']['x'].loc[index_loc])**2) + ((x_center[1] - df['nose']['y'].loc[index_loc])**2))
        distance_to_l_ear = np.sqrt(((x_center[0] - df['left ear']['x'].loc[index_loc])**2) + ((x_center[1] - df['left ear']['y'].loc[index_loc])**2))
        distance_to_r_ear = np.sqrt(((x_center[0] - df['right ear']['x'].loc[index_loc])**2) + ((x_center[1] - df['right ear']['y'].loc[index_loc])**2))
    
    ## and changing the orientation variable to 'oriented' if the distance to nose is indeed the lowest value ##
    
    if distance_to_nose == np.min([distance_to_nose, distance_to_l_ear, distance_to_r_ear]):
        orientation = 'oriented'
        
    return orientation

