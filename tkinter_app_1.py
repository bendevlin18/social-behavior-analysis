
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


root = Tk()
root.wm_geometry('1280x720')
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

header_frame = LabelFrame(root, padx = 5, pady = 5)
header_frame.grid(padx = 10, pady = 10, sticky='nsew')

direc_frame = LabelFrame(root, padx = 5, pady = 5)
direc_frame.grid(padx = 10, pady = 10, sticky='nsew')

img_frame = LabelFrame(root, padx = 5, pady = 5)
img_frame.grid(padx = 10, pady = 10, sticky='nsew')

mp4_frame = LabelFrame(root, padx = 5, pady = 5)
mp4_frame.grid(padx = 10, pady = 10, sticky='nsew')

tab_frame = LabelFrame(root, padx = 5, pady = 5)
tab_frame.grid(padx = 10, pady = 10, sticky='nsew')


root.title('Home Page')
# root.iconbitmap('representative_cleaned_hemisphere_frontal_cortex_segments_iGT_icon.ico')
font_style_big = tkFont.Font(family="Lucida Grande", size=50)
font_style_small = tkFont.Font(family="Lucida Grande", size=35)

main_page_label = Label(header_frame, text = 'Welcome to the main page', font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

def direc_btn():
    directory = filedialog.askdirectory(initialdir = os.getcwd(), title = 'Select the analysis directory')
    global main_dir
    main_dir = directory
    main_direc_readout = Label(direc_frame, text = directory, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')

main_direc_label = Label(direc_frame, text = 'Enter the main analysis directory here').grid(row = 1, column = 0, sticky='nsew')
main_direc_button = Button(direc_frame, text = '...', command = direc_btn).grid(row = 1, column = 1, sticky='nsew')

def video_dir_btn():
    global v_location
    v_location = filedialog.askdirectory(initialdir = os.getcwd(), title = 'Select the video directory')
    main_video_location_readout = Label(mp4_frame, text = v_location, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')

main_video_loc_label = Label(mp4_frame, text = 'Navigate to the master directory with all the videos').grid(row = 1, column = 0, sticky='nsew')
main_video_loc_button = Button(mp4_frame, text = '...', command = video_dir_btn).grid(row = 1, column = 1, sticky='nsew')

def annotate_window():
    annotate = Toplevel()
    annotate.wm_geometry('1280x720')
    annotate.title('Annotate Chamber Coordinates')
    if not os.path.exists(main_dir + '\\coordinates'):
        os.mkdir(main_dir + '\\coordinates')
    cap = cv2.VideoCapture(v_location + '\\' + os.listdir(v_location)[0])
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
            np.savetxt(main_dir + '\\coordinates\\' + region, verts)

        polygon = PolygonSelector(ax, onselect)

        plt.show()

    locations = ['x_chamber', 'y_chamber', 'left_side', 'right_side', 'middle']

    for location in locations:
        selectROI(location)
    button_quit = Button(annotate, text = 'All Done! Exit This Step', command = annotate.destroy).pack()

main_annotate_btn = Button(tab_frame, text = 'Step 1: Annotate Video Frame', command = annotate_window).pack(side = 'left')



def create_coord_window():
    
    left_side = np.loadtxt(main_dir + '\\coordinates\\left_side')
    x_chamber = np.loadtxt(main_dir + '\\coordinates\\x_chamber')
    y_chamber = np.loadtxt(main_dir + '\\coordinates\\y_chamber')
    right_side = np.loadtxt(main_dir + '\\coordinates\\right_side')
    middle = np.loadtxt(main_dir + '\\coordinates\\middle')
    
    ## calculate the padded interaction zone, that is 40 pixels larger in every direction (than the base of the chamber)

    x_outer = Polygon(Polygon(x_chamber).buffer(40).exterior).exterior.coords.xy
    y_outer = Polygon(Polygon(y_chamber).buffer(40).exterior).exterior.coords.xy
    x_center = np.mean(x_outer[0]), np.mean(x_outer[1])
    y_center = np.mean(y_outer[0]), np.mean(y_outer[1])
    
    x_zone = Polygon(Polygon(x_chamber).buffer(40).exterior)
    y_zone = Polygon(Polygon(y_chamber).buffer(40).exterior)
    
    global possible_places
    possible_places = {'x_zone': x_zone, 'y_zone': y_zone, 'left_side': left_side, 'middle': middle, 'right_side': right_side}
    global extra_coords
    extra_coords = {'x_outer': x_outer, 'x_center': x_center, 'y_outer': y_outer, 'y_center': y_center, 'x_chamber': x_chamber, 'y_chamber': y_chamber}
    
    if possible_places:
        coordinate_output = Label(root, text = """
        
        Coordinates Imported Successfully!
        If you would like to adjust them, just redo step 1!"""
        
        ).grid(row = 0, column = 0, sticky='nsew')
        coordinates = [possible_places, extra_coords]

        cap = cv2.VideoCapture(v_location + '\\' + os.listdir(v_location)[0])
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



        fig, ax = plt.subplots(1,1,figsize=(5,3))
        ax.plot(coordinates[0]['left_side'][:,0], coordinates[0]['left_side'][:,1], color = 'red', linewidth = 4)
        ax.plot(coordinates[0]['right_side'][:,0], coordinates[0]['right_side'][:,1], color = 'purple', linewidth = 4)
        ax.plot(coordinates[1]['x_chamber'][:,0], coordinates[1]['x_chamber'][:,1], color = 'blue', linewidth = 4)
        ax.plot(coordinates[1]['y_chamber'][:,0], coordinates[1]['y_chamber'][:,1], color = 'green', linewidth = 4)
        ax.plot(coordinates[1]['x_outer'][0], coordinates[1]['x_outer'][1], color = 'blue', linewidth = 4)
        ax.plot(coordinates[1]['y_outer'][0], coordinates[1]['y_outer'][1], color = 'green', linewidth = 4)
        ax.plot(coordinates[1]['x_center'][0], coordinates[1]['x_center'][1], 'bo')
        ax.plot(coordinates[1]['y_center'][0], coordinates[1]['y_center'][1], 'go')
        ax.imshow(frame)
        plt.xlim(0,1280)
        plt.ylim(0,720)
        plt.show()

main_coord_import_btn = Button(tab_frame, text = 'Step 2: Import & Show Coordinate Annotations', command = create_coord_window).pack(side = 'left')

progress_var = DoubleVar()
heatmap_progress.grid()
def show_heatmaps():
    coordinates = [possible_places, extra_coords]

    trial_frames = (450,7500)
    videos = os.listdir(main_dir + '\\csv_outputs')
    if not os.path.exists(main_dir + '\\heatmaps'):
        os.mkdir(main_dir + '\\heatmaps')

    for i in range(len(videos)):

        df = pd.read_csv(main_dir + '\\csv_outputs\\' + videos[i], header = [1, 2])

        fig, ax = plt.subplots(1,1,figsize=(14.4,9.6))
        ax.plot(df['nose']['x'].loc[trial_frames[0]:trial_frames[1]], df['nose']['y'].loc[trial_frames[0]:trial_frames[1]], c='k', alpha=.3, marker='o', linestyle='None')
        ax.plot(coordinates[0]['left_side'][:,0], coordinates[0]['left_side'][:,1], color = 'red', linewidth = 4)
        ax.plot(coordinates[0]['right_side'][:,0], coordinates[0]['right_side'][:,1], color = 'purple', linewidth = 4)
        ax.plot(coordinates[1]['x_chamber'][:,0], coordinates[1]['x_chamber'][:,1], color = 'blue', linewidth = 4)
        ax.plot(coordinates[1]['y_chamber'][:,0], coordinates[1]['y_chamber'][:,1], color = 'green', linewidth = 4)
        ax.plot(coordinates[1]['x_outer'][0], coordinates[1]['x_outer'][1], color = 'blue', linewidth = 4)
        ax.plot(coordinates[1]['y_outer'][0], coordinates[1]['y_outer'][1], color = 'green', linewidth = 4)
        ax.plot(coordinates[1]['x_center'][0], coordinates[1]['x_center'][1], 'bo')
        ax.plot(coordinates[1]['y_center'][0], coordinates[1]['y_center'][1], 'go')
        plt.xlim(0,1280)
        plt.ylim(0,720)
        plt.savefig(main_dir + '\\heatmaps\\' + videos[i] + '.png')
        plt.close()

        heatmap_output = Label(root, text = 'All Finished! Heatmaps are now saved in the heatmaps folder in the main analysis directory').grid(row = 0, column = 0, sticky='nsew')


main_heatmap_generator_btn = Button(tab_frame, text = 'Step 3: Create Heatmap / Validate Labels (OPTIONAL)', command = show_heatmaps).pack(side = 'left')


def prep_time_df():

    
    global df_times
    df_times = pd.read_excel(main_dir + '\\start_times.xlsx')
    
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

    times_output = Label(root, text = """
    
    Time Dataframe Imported Successfully!
    Now you can calculate investigation times!"""
    
    ).grid(row = 0, column = 0, sticky='nsew')

    print(df_times)


main_time_df_import_btn = Button(tab_frame, text = 'Step 4: Import and Format Time Dataframe', command = prep_time_df).pack(side = 'left')

###################################### DEFINING FUNCTIONS THAT WILL BE IMPORTED WITH THE PACKAGE ONCE LIVE ###############################################################


def check_coords(coords):

    x = []

    for i in range(len(list(possible_places.values()))):   
        pt = Point(coords)
        if isinstance(list(possible_places.values())[i], Polygon):
            polygon = list(possible_places.values())[i]
        else:
            polygon = Polygon(list(map(tuple, list(possible_places.values())[i])))
        x = np.append(x, polygon.contains(pt))

    return x 



def check_climbing(df, coords):

    state = ['climbing'] * len(df)
    z = -1
    
    for index, val in df.iterrows():
        z = z + 1
        distance_1 = np.sqrt(((df['left ear']['x'].loc[index] - df['nose']['x'].loc[index])**2) + ((df['left ear']['y'].loc[index] - df['nose']['y'].loc[index])**2))
        distance_2 = np.sqrt(((df['right ear']['x'].loc[index] - df['left ear']['x'].loc[index])**2) + ((df['right ear']['y'].loc[index] - df['left ear']['y'].loc[index])**2))
        distance_3 = np.sqrt(((df['nose']['x'].loc[index] - df['right ear']['x'].loc[index])**2) + ((df['nose']['y'].loc[index] - df['right ear']['y'].loc[index])**2))

        if np.sum(distance_1, distance_2, distance_3) > 5:
            state[z] = 'not_climbing'

def check_orientation_single(df, index_loc):

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
        
    if distance_to_nose == np.min([distance_to_nose, distance_to_l_ear, distance_to_r_ear]):
        orientation = 'oriented'
        
    return orientation



###########################################################################################################################################################################


def calculate_investigation_times(behavior_type = 'Social', bodypart = 'nose', video_suffix = 'DLC_resnet50_social_behavior_allMay27shuffle1_250000.csv'):

    final_dict = {}
    csv_direc = main_dir + '\\csv_outputs\\'


    for i in range(len(df_times)):
 
        ### first we will want to get the right dataframe, so we should import it based on the df_times location and clean it

        df = pd.read_csv(csv_direc + df_times['VideoName'][i] + video_suffix, header = [1, 2], index_col = 0)
        bodyparts = np.unique(df.columns.get_level_values(0))        

        int_df = df.loc[df_times['Start' + behavior_type + 'Frames'][i]:df_times['Stop' + behavior_type + 'Frames'][i]]

        ### need to lose frame information and return it back to 0:end         
        int_df.reset_index(drop=True, inplace = True) 

        arr = np.zeros(shape = (len(int_df), len(bodyparts), len(possible_places)))

        ### now we should check the coordinates of each bodypart in each frame
        for row in range(len(int_df)):
            for j in range(len(bodyparts)):
                arr[row][j] = check_coords(int_df[bodyparts[j]][['x', 'y']].loc[row].values)

        ### set which patterns mean x vs y investigation, only for the first three bodyparts (nose and ears, cuz we don't care about tail base yet)
        if bodypart == 'nose':
            x_inv = np.array([[1., 0., 1., 0., 0.]])
            y_inv = np.array([[0., 1., 0., 0., 1.]])

        if bodypart == 'nose_and_ears':
            x_inv = np.array([[1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.]])
            y_inv = np.array([[0., 1., 0., 0., 1.], [1., 0., 1., 0., 0.], [1., 0., 1., 0., 0.]])           


        ### now we want to check each frame in our array, and create a frame_val array that holds info about where the mouse's head was detected
        z = -1
        frame_val = np.zeros(shape = len(arr), dtype = 'object')
        for frame in range(len(arr)):
            z = z + 1
            comparison_x = arr[frame][0:1] == x_inv
            comparison_y = arr[frame][0:1] == y_inv

            if comparison_x.all() == True:
                if check_orientation_single(int_df, z) == 'oriented':
                    frame_val[z] = 'X Investigation'
                elif check_orientation_single(int_df, z) == 'not_oriented':
                    frame_val[z] = 'X Close'
            elif comparison_y.all() == True:
                if check_orientation_single(int_df, z) == 'oriented':
                    frame_val[z] = 'Y Investigation'
                elif check_orientation_single(int_df, z) == 'not_oriented':
                    frame_val[z] = 'Y Close'
            else:
                frame_val[z] = 'Somewhere else'
                
        print(np.unique(frame_val, return_counts = True))
        

        x_invest = list(frame_val).count('X Investigation') 
        x_close = list(frame_val).count('X Close')  
        y_invest = list(frame_val).count('Y Investigation') 
        y_close = list(frame_val).count('Y Close')  
        somewhere_else = list(frame_val).count('Somewhere else')  
                
        final_frame_counts = [somewhere_else, x_invest, y_invest, x_close, y_close]

        final_dict[df_times['VideoName'][i]] = final_frame_counts


        global output_df
        output_df = pd.DataFrame(final_dict, index = ['Somewhere else','X Investigation', 'Y Investigation', 'X Close', 'Y Close']).T
        
        output_df['type'] = [behavior_type] * len(output_df)
        output_df.reset_index(inplace = True)
        output_df.set_index(['index', 'type'], inplace = True)

        print('Just finished video' + str(i + 1) + 'of' + str(len(df_times)))
    output_df.to_csv('\\output.csv')

main_calculate_invest_btn = Button(tab_frame, text = 'Step 5: Calculate Investigation Times!', command = calculate_investigation_times).pack(side = 'left')


def convert_to_secs():
    
    frameRate = np.zeros(shape = len(df_times))
    i = -1
    global new_df
    new_df = pd.DataFrame(columns = output_df.columns, index = output_df.index)
    for index, values in df_times.iterrows():
        i = i + 1
        cap = cv2.VideoCapture(v_location + '\\' + values['VideoName'] + '.mp4')
        frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
        index = output_df.index[i]
        new_df.loc[index[0], index[1]] = (output_df.loc[index[0]].loc[index[1]] / frameRate[i]).values
    else:
        pass
        
    new_df.to_csv(main_dir + '\\adjusted_output.csv')







root.mainloop()