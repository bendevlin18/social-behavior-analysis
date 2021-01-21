
######## Importing ALLL the packages that we will need ########

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
from analysis_functions_master import *
from tqdm import tqdm

####### Creating all of the frames for the GUI #########

root = Tk()
root.wm_geometry('1600x900')
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

######### Setting Root Title and font parameters #########

root.title('Home Page')
font_style_big = tkFont.Font(family="Lucida Grande", size=50)
font_style_small = tkFont.Font(family="Lucida Grande", size=35)
main_page_label = Label(header_frame, text = 'Welcome to the main page', font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

######### Button for grabbing the main analysis directory #########

def direc_btn():

    ## I have built this analysis GUI such that there is a single main directory that houses the videos, as well as the output CSV files from DLC ##
    ## Everything stems from this folder, and this directory location is chosen by the user and saved as a global variable, as everything will be saved within this directory ##

    global main_dir
    directory = filedialog.askdirectory(initialdir = os.getcwd(), title = 'Select the analysis directory')
    main_dir = directory
    main_direc_readout = Label(direc_frame, text = directory, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')

main_direc_label = Label(direc_frame, text = 'Enter the main analysis directory here').grid(row = 1, column = 0, sticky='nsew')
main_direc_button = Button(direc_frame, text = '...', command = direc_btn).grid(row = 1, column = 2, sticky='nsew')

######### Button for grabbing the video directory ##########

def video_dir_btn():

    ## now we are navigating to the subdirectory that has the video mp4s saved in it ##
    ## this is saved also a global variable, which will allow us to extract these videos for creating heatmaps or plotting frames for annotation ##

    global v_location
    v_location = filedialog.askdirectory(initialdir = os.getcwd(), title = 'Select the video directory')
    main_video_location_readout = Label(mp4_frame, text = v_location, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')

main_video_loc_label = Label(mp4_frame, text = 'Navigate to the master directory with all the videos').grid(row = 1, column = 0, sticky='nsew')
main_video_loc_button = Button(mp4_frame, text = '...', command = video_dir_btn).grid(row = 1, column = 2, sticky='nsew')


######### Button for Step 1, which is to annotate the video and define the ROIs #########

def annotate_window():

    ## creating a directory to save the numpy text files that contain coordinates for the annotations made at this step ##

    if not os.path.exists(main_dir + '\\coordinates'):
        os.mkdir(main_dir + '\\coordinates')
    frame = grab_video_frame(v_location)

    ## list of the 5 locations I have included for analysis ##
    ## I would like in the future to have this customizable in the GUI ##
    locations = ['x_chamber', 'y_chamber', 'left_side', 'right_side', 'middle']

    ## loops through the locations in the list, and allow the user to save a polygon annotation as a np text file for each zone ##
    for location in locations:
        selectROI(location, frame, main_dir)
    annotate_output = Label(root, text = 'All Done! Coordinates Saved in Coordinates Directory', font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

main_annotate_btn = Button(tab_frame, text = 'Step 1: Annotate Video Frame', command = annotate_window).pack(side = 'left')


########## Button for Step 2, which imports the coordinates and displays them over a representative frame of the video #########

def create_coord_window():

    ## loading in the 5 annotation zones that were created in the last step ##
    
    left_side = np.loadtxt(main_dir + '\\coordinates\\left_side')
    x_chamber = np.loadtxt(main_dir + '\\coordinates\\x_chamber')
    y_chamber = np.loadtxt(main_dir + '\\coordinates\\y_chamber')
    right_side = np.loadtxt(main_dir + '\\coordinates\\right_side')
    middle = np.loadtxt(main_dir + '\\coordinates\\middle')
    
    ## calculate the padded interaction zone, that is 40 pixels larger in every direction (than the base of the chamber) ##
    ## In the future I would like to make the padding size flexible based on user input if we think it might need to change ##

    x_outer = Polygon(Polygon(x_chamber).buffer(40).exterior).exterior.coords.xy
    y_outer = Polygon(Polygon(y_chamber).buffer(40).exterior).exterior.coords.xy
    x_center = np.mean(x_outer[0]), np.mean(x_outer[1])
    y_center = np.mean(y_outer[0]), np.mean(y_outer[1])
    
    x_zone = Polygon(Polygon(x_chamber).buffer(40).exterior)
    y_zone = Polygon(Polygon(y_chamber).buffer(40).exterior)

    ## these two global variables hold the coordinates for the annotated zones (possible_places) ##
    ## as well as the calculated zones/points necessary for analysis (extra_coords) ##
    
    global possible_places
    possible_places = {'x_zone': x_zone, 'y_zone': y_zone, 'left_side': left_side, 'middle': middle, 'right_side': right_side}
    global extra_coords
    extra_coords = {'x_outer': x_outer, 'x_center': x_center, 'y_outer': y_outer, 'y_center': y_center, 'x_chamber': x_chamber, 'y_chamber': y_chamber}
    
    if possible_places:
        coordinate_output = Label(root, text = """
        
        Coordinates Imported Successfully!
        If they need adjusted, just redo step 1!"""
        
        , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')
        coordinates = [possible_places, extra_coords]

        plot_coordinates_frame(grab_video_frame(v_location), coordinates)

main_coord_import_btn = Button(tab_frame, text = 'Step 2: Import & Show Coordinate Annotations', command = create_coord_window).pack(side = 'left')


######### Button for Step 3, creating heatmaps for every video. This is optional and still in alpha #########

def show_heatmaps():
    coordinates = [possible_places, extra_coords]

    ## extracting the DLC csv output from the csv_output folder that MUST be in the master analysis directory ##
    ## creating a heatmaps folder to save the heatmaps for each video ##
    csv_file_list = os.listdir(os.path.join(main_dir, 'csv_outputs'))
    videos = []
    for csv_file in csv_file_list:
        if '.csv' in file_list:
            videos = np.append(videos, csv_file)
    if not os.path.exists(os.path.join(main_dir, 'heatmaps')):
        os.mkdir(os.path.join(main_dir, 'heatmaps'))

    ## loop through the lis of videos and 
    for i in range(len(videos)):

        df = pd.read_csv(os.path.join(main_dir, 'csv_outputs', videos[i]), header = [1, 2])

        trial_frames = (df_times['StartSocialFrames'][i] + 30, df_times['StopSocialFrames'][i] + 30)

        plot_heatmap(coordinates, df, trial_frames)

        plt.savefig(os.path.join(main_dir, 'heatmaps', videos[i], '.png'))
        plt.close()

        heatmap_output = Label(root, text = """
        
        All Finished! 
        Heatmaps are now saved in the heatmaps folder in the main analysis directory"""
        
        , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')


main_heatmap_generator_btn = Button(tab_frame, text = 'Step 3: Create Heatmap / Validate Labels (OPTIONAL)', command = show_heatmaps).pack(side = 'left')


######### Button for Step 4, preparing the time dataframe #########


def prep_time_df():

    ## function for reading in the time dataframe. this is a separate excel/csv that contains the time that the animal was actually placed in the chamber (this is not always the very start of the video)
    global df_times
    df_times = pd.read_excel(os.path.join(main_dir, 'start_times.xlsx'))
    
    ## adjusting the timedf with time_df function to calculate start and stop frames
    df_times = time_df(df_times, v_location)

    times_output = Label(root, text = """
    
    Time Dataframe Imported Successfully!
    Now you can calculate investigation times!"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

    print(df_times)


main_time_df_import_btn = Button(tab_frame, text = 'Step 4: Import and Format Time Dataframe', command = prep_time_df).pack(side = 'left')




######### Button for Step 5, which is the main calculation and analysis based on ROIs #########

def calculate_investigation_times(behavior_type = 'Social', bodypart = 'nose', video_suffix = 'DLC_resnet50_social_behavior_allMay27shuffle1_250000.csv'):

    final_dict = {}
    csv_direc = os.path.join(main_dir, 'csv_outputs')


    for i in range(len(df_times)):
 
        ### first we will want to get the right dataframe, so we should import it based on the df_times location and clean it

        df = pd.read_csv(csv_direc + df_times['VideoName'][i] + video_suffix, header = [1, 2], index_col = 0)
        bodyparts = np.unique(df.columns.get_level_values(0))        

        int_df = df.loc[df_times['Start' + behavior_type + 'Frames'][i]:df_times['Stop' + behavior_type + 'Frames'][i]]

        ### need to lose frame information and return it back to 0:end         
        int_df.reset_index(drop=True, inplace = True) 

        arr = np.zeros(shape = (len(int_df), len(bodyparts), len(possible_places)))

        ### now we should check the coordinates of each bodypart in each frame
        

        for row in tqdm(range(len(int_df))):
            for j in range(len(bodyparts)):
                arr[row][j] = check_coords(int_df[bodyparts[j]][['x', 'y']].loc[row].values, possible_places)

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
        for frame in tqdm(range(len(arr))):
            z = z + 1
            comparison_x = arr[frame][0:1] == x_inv
            comparison_y = arr[frame][0:1] == y_inv

            if comparison_x.all() == True:
                if check_orientation_single(int_df, z, extra_coords) == 'oriented':
                    frame_val[z] = 'X Investigation'
                elif check_orientation_single(int_df, z, extra_coords) == 'not_oriented':
                    frame_val[z] = 'X Close'
            elif comparison_y.all() == True:
                if check_orientation_single(int_df, z, extra_coords) == 'oriented':
                    frame_val[z] = 'Y Investigation'
                elif check_orientation_single(int_df, z, extra_coords) == 'not_oriented':
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

        print('Just finished Video ' + str(i + 1) + ' of ' + str(len(df_times)))
    output_df.to_csv(os.path.join(main_dir, 'output.csv'))

    invest_output = Label(root, text = """
    
    All Investigation times calculated!
    Frame Values placed in output.csv!"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

main_calculate_invest_btn = Button(tab_frame, text = 'Step 5: Calculate Investigation Times!', command = calculate_investigation_times).pack(side = 'left')



def convert_to_secs():

    global new_df

    if os.path.join(main_dir, 'output.csv'):
        output_df = pd.read_csv(os.path.join(main_dir, 'output.csv'), index_col = 0)

    new_df = pd.DataFrame(columns = output_df.columns, index = output_df.index)

    frameRate = np.zeros(shape = len(df_times))
    i = -1
    for index, values in df_times.iterrows():
        i = i + 1
        cap = cv2.VideoCapture(v_location + '\\' + values['VideoName'] + '.mp4')
        frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
        index = output_df.index[i]
        new_df.loc[index][1:6] = (output_df.loc[index][1:6] / frameRate[i]).values
    else:
        pass
    new_df['type'] = output_df['type']
    new_df.to_csv(main_dir + '\\adjusted_output.csv')
    secs_output = Label(root, text = """
    
    Investigation times converted to seconds!
    Output placed in adjusted_output.csv"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')


convert_2_secs_btn = Button(tab_frame, text = 'Step 6: Convert Investigation times to Seconds', command = convert_to_secs).pack(side = 'left')




root.mainloop()