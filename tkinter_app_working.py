
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
from tkinter import ttk, simpledialog
from PIL import ImageTk, Image
#from analysis_functions_master import *
from tqdm import tqdm

# #new files
from function_folder.annotation import *
from function_folder.video_making import *
from function_folder.calc_invest_times import *
from function_folder.smoothing import *
from function_folder.utils import *
from function_folder.time_df import *
from function_folder.heat_map import *

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

csv_output_folder_frame = LabelFrame(root, padx = 5, pady = 5)
csv_output_folder_frame.grid(padx = 10, pady = 10, sticky='nsew')

mp4_frame = LabelFrame(root, padx = 5, pady = 5)
mp4_frame.grid(padx = 10, pady = 10, sticky='nsew')


df_times_frame = LabelFrame(root, padx = 5, pady = 5)
df_times_frame.grid(padx = 10, pady = 10, sticky='nsew')

tab_frame = LabelFrame(root, padx = 5, pady = 5)
tab_frame.grid(padx = 10, pady = 10, sticky='nsew')

######### Setting Root Title and font parameters #########

root.title('Home Page')
font_style_big = tkFont.Font(family="Lucida Grande", size=50)
font_style_small = tkFont.Font(family="Lucida Grande", size=5)
main_page_label = Label(header_frame, text = 'Main page', font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

######### Button for grabbing the main analysis directory #########

def direc_btn():

    ## I have built this analysis GUI such that there is a single main directory that houses the videos, as well as the output CSV files from DLC ##
    ## Everything stems from this folder, and this directory location is chosen by the user and saved as a global variable, as everything will be saved within this directory ##

    global main_dir
    directory = filedialog.askdirectory(initialdir = os.getcwd(), title = 'Select the analysis directory')
    main_dir = directory
    main_direc_readout = Label(direc_frame, text = directory, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')
    os.chdir(main_dir)

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



def csv_folder_btn():

    ##  ##
    ##  ##

    global csv_output_folder
    csv_output_folder = filedialog.askdirectory(initialdir = os.getcwd(), title = 'Select the csv file directory')
    csv_output_folder_location = Label(csv_output_folder_frame, text = csv_output_folder, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')

df_times_loc_label = Label(csv_output_folder_frame, text = 'Navigate to the folder with the csv coordinate outputs').grid(row = 1, column = 0, sticky='nsew')
df_times_loc_button = Button(csv_output_folder_frame, text = '...', command = csv_folder_btn).grid(row = 1, column = 2, sticky='nsew')

def df_time_file_btn():

    ##  ##
    ##  ##

    global df_times_filename
    df_times_filename = filedialog.askopenfilename(initialdir = os.getcwd(), title = 'Select the excel file with start times')
    start_times_location_readout = Label(df_times_frame, text = df_times_filename, bg = 'black', fg = 'white', borderwidth = 5).grid(row = 1, column = 1, sticky='nsew')

df_times_loc_label = Label(df_times_frame, text = 'Navigate to the file with the start times').grid(row = 1, column = 0, sticky='nsew')
df_times_loc_button = Button(df_times_frame, text = '...', command = df_time_file_btn).grid(row = 1, column = 2, sticky='nsew')





######### Button for Step 1, which is to annotate the video and define the ROIs #########

def annotate_window():

    ## creating a directory to save the numpy text files that contain coordinates for the annotations made at this step ##

    if not os.path.exists(os.path.join(main_dir, 'coordinates')):
        os.mkdir(os.path.join(main_dir, 'coordinates'))
    frame = grab_video_frame(v_location)

    ## list of the 5 locations I have included for analysis ##
    ## I would like in the future to have this customizable in the GUI ##
    locations = ['x_chamber', 'y_chamber', 'left_side', 'right_side', 'middle']

    ## loops through the locations in the list, and allow the user to save a polygon annotation as a np text file for each zone ##
    for location in locations:
        selectROI(location, frame, main_dir)
    annotate_output = Label(root, text = 'All Done! Coordinates Saved in Coordinates Directory', font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

main_annotate_btn = Button(tab_frame, text = '1: Annotate', command = annotate_window).pack(side = 'left')


########## Button for Step 2, which imports the coordinates and displays them over a representative frame of the video #########

def create_coord_window():

    ## loading in the 5 annotation zones that were created in the last step ##
    
    coord_path = os.path.join(main_dir, 'coordinates')

    left_side = np.loadtxt(os.path.join(coord_path, 'left_side'))
    x_chamber = np.loadtxt(os.path.join(coord_path, 'x_chamber'))
    y_chamber = np.loadtxt(os.path.join(coord_path, 'y_chamber'))
    right_side = np.loadtxt(os.path.join(coord_path, 'right_side'))
    middle = np.loadtxt(os.path.join(coord_path, 'middle'))
    
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

main_coord_import_btn = Button(tab_frame, text = '2: Coordinate', command = create_coord_window).pack(side = 'left')


######### Button for Step 3, preparing the time dataframe #########

def prep_time_df():

    ## function for reading in the time dataframe. this is a separate excel/csv that contains the time that the animal was actually placed in the chamber (this is not always the very start of the video)
    global df_times
    
    df_times = pd.read_csv(os.path.join(main_dir, df_times_filename))

    ## adjusting the timedf with time_df function to calculate start and stop frames
    df_times = time_df(df_times, v_location)

    times_output = Label(root, text = """
    
    Time Dataframe Imported Successfully!
    Now you can calculate investigation times!"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

    print(df_times)


main_time_df_import_btn = Button(tab_frame, text = '3: Time', command = prep_time_df).pack(side = 'left')

######### Button to process and/or import preprocessed CSVs #########

def preprocess_df():

    print('Now preprocessing and smoothing the raw DLC output')
    global processed_csv_output_folder

    if not os.path.exists(os.path.join(main_dir, 'smoothed_csv_output')):
        os.mkdir(os.path.join(main_dir, 'smoothed_csv_output'))
        
        processed_csv_output_folder = os.path.join(main_dir, 'smoothed_csv_output')

        count = 0
        for file in os.listdir(csv_output_folder):
            count += 1
            process_csv(pd.read_csv(os.path.join(csv_output_folder, file), header = [1, 2])).to_csv(os.path.join(processed_csv_output_folder, file))
            print('Finished file ', str(count), ' of', str(len(os.listdir(csv_output_folder))))
    elif os.path.exists(os.path.join(main_dir, 'smoothed_csv_output')):
        processed_csv_output_folder = os.path.join(main_dir, 'smoothed_csv_output')

    print('CSVs smoothed and imported!')

preprocess_csv_btn = Button(tab_frame, text = '4: Process CSVs', command = preprocess_df).pack(side = 'left')






######### Button for Step 4, creating heatmaps for every video. This is optional and still in alpha #########

def show_heatmaps():
    coordinates = [possible_places, extra_coords]

    ## extracting the DLC csv output from the csv_output folder that is in the master analysis directory ##
    ## creating a heatmaps folder to save the heatmaps for each video ##
    videos = os.listdir(os.path.join(main_dir, processed_csv_output_folder))

    behavior_type = simpledialog.askstring('Choose behavior type', 'Which behavior would you like to analyze? (Social or Novel)')

    if not os.path.exists(os.path.join(main_dir, behavior_type + '_heatmaps')):
        os.mkdir(os.path.join(main_dir, behavior_type + '_heatmaps'))


    ## loop through the list of videos and plot a heatmap on them
    for i in range(len(videos)):
        if 'csv' in videos[i]:
            df = pd.read_csv(os.path.join(main_dir, processed_csv_output_folder, videos[i]), header = [0, 1])

            trial_frames = (df_times['Start' + behavior_type + 'Frames'][i] + 30, df_times['Stop' + behavior_type + 'Frames'][i] + 30)

            plot_heatmap_dark(coordinates, df, trial_frames)

            plt.savefig(os.path.join(main_dir, behavior_type + '_heatmaps', videos[i] + '.png'))
            print(videos[i] + ' heatmap saved!')
            plt.close()

        heatmap_output = Label(root, text = """
        
        All Finished! 
        Heatmaps are now saved in the heatmaps folder in the main analysis directory"""
        
        , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')


main_heatmap_generator_btn = Button(tab_frame, text = '4: Heatmaps', command = show_heatmaps).pack(side = 'left')



######### Button for Step 5, which is the main calculation and analysis based on ROIs #########

def calc_investigation_times(bodypart = 'nose'):

    global video_suffix
    global behavior_type
    #video_suffix = simpledialog.askstring('DLC_resnet50_social_behavior_allMay27shuffle1_250000', 'What is the DLC suffix?')
    #trying to get video_suffix without needing to ask for it
    csv_direc = os.path.join(main_dir, processed_csv_output_folder)
    first_vid = os.listdir(csv_direc)[0]
    video_suffix_start = first_vid.index("DLC")
    video_suffix = first_vid[video_suffix_start: len(first_vid)]

    frame_val_dir = os.path.join(main_dir, "frame_values")
    if not os.path.exists(frame_val_dir):
        os.mkdir(frame_val_dir)

    for i in range(len(df_times)):
 
        ### first we will want to get the right dataframe, so we should import it based on the df_times location and clean it

        df = pd.read_csv(os.path.join(csv_direc, df_times['VideoName'][i] + video_suffix), header = [0, 1], index_col = 0).dropna()
        frame_val = calculate_investigation_times(df, possible_places, extra_coords)

        #makes a new folder and saves frame_vals
        frame_filepath = os.path.join(frame_val_dir, df_times['VideoName'][i]) + "_frame_val.csv"
        pd.DataFrame(frame_val).to_csv(frame_filepath)

        print('Just finished Video ' + str(i + 1) + ' of ' + str(len(df_times)))

    invest_output = Label(root, text = """
    
    All Investigation times calculated!
    Frame Values placed in output.csv!"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')

main_calculate_invest_btn = Button(tab_frame, text = '5: Calculate', command = calc_investigation_times).pack(side = 'left')



def convert_to_secs():

    global new_df

    #behavior_type = simpledialog.askstring('Choose behavior type', 'Which behavior would you like to analyze? (Social or Novel)')

    if os.path.join(main_dir, "all" + '_output.csv'):
        output_df = pd.read_csv(os.path.join(main_dir, "all" + '_output.csv'), index_col = 0)

    new_df = pd.DataFrame(columns = output_df.columns, index = output_df.index)

    frameRate = np.zeros(shape = len(df_times))
    i = -1
    for index, values in df_times.iterrows():
        i = i + 1
        cap = cv2.VideoCapture(os.path.join(v_location, values['VideoName'] + '.mp4'))
        frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
        index = output_df.index[i]
        new_df.loc[index][1:6] = (output_df.loc[index][1:6] / frameRate[i]).values
    else:
        pass
    new_df['type'] = output_df['type']
    new_df.to_csv(os.path.join(main_dir, "all" + '_adjusted_output.csv'))
    secs_output = Label(root, text = """
    
    Investigation times converted to seconds!
    Output placed in adjusted_output.csv"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')


convert_2_secs_btn = Button(tab_frame, text = '6: to Secs', command = convert_to_secs).pack(side = 'left')



##### Button for step 7 - creating labelled frames that can be stitched into a labelled video #####


def export_frames_with_label():
    video_to_label = filedialog.askopenfilename(initialdir = v_location, title = 'Select a video that you want to label!')
    video_to_label_path = os.path.join(v_location, video_to_label)
    csv_to_label = filedialog.askopenfilename(initialdir = processed_csv_output_folder, title = 'Select the corresponding csv file')
    csv_direc = os.path.join(main_dir, processed_csv_output_folder)
    df = pd.read_csv(os.path.join(csv_direc, csv_to_label), header = [0, 1]).dropna().reset_index(drop = True)
   
    #implement a way to load investigation times from frame_values
    frame_values_folder = os.path.join(main_dir, 'frame_values')
    frame_val_to_label = filedialog.askopenfilename(initialdir = frame_values_folder, title = 'Select the corresponding frame_values file')
    invest_times = pd.read_csv(os.path.join(frame_values_folder, frame_val_to_label))


    export_labelled_frames(df, video_to_label_path, frame_val = invest_times['0'].values, output_dir = os.path.join(main_dir, 'labelled_frames'))
    labelled_frames_output = Label(root, text = """
    
    Frames have been labelled!
    They are stored in the labelled frames directory in main analysis folder"""
    
    , font = font_style_big).grid(row = 0, column = 0, sticky='nsew')


    ############# use FFmpeg to turn frames into movie, then delete the frames folder ##############

export_labelled_frames_btn = Button(tab_frame, text = '7: Label', command = export_frames_with_label).pack(side = 'left')



def build_video_from_frames():

    os.listdir(os.path.join(main_dir, 'labelled_frames'))[1]

    ffmpeg_make_video(main_dir, os.path.join(main_dir, 'labelled_frames'), vname = 'testing')

make_video_from_frames_btn = Button(tab_frame, text = 'Make video from frames', command = build_video_from_frames).pack(side = 'left')

##### Button for step8 - calculating total distance travelled for each trial

def total_distance_travelled():

    behavior_type = simpledialog.askstring('Choose behavior type', 'Which behavior would you like to analyze? (Social or Novel)')
    #video_suffix = simpledialog.askstring('DLC_resnet50_social_behavior_allMay27shuffle1_250000', 'What is the DLC suffix?')
    #trying to get video_suffix without needing to ask for it
    csv_direc = os.path.join(main_dir, processed_csv_output_folder)
    first_vid = os.listdir(csv_direc)[0]
    video_suffix_start = first_vid.index("DLC")
    video_suffix = first_vid[video_suffix_start: len(first_vid)]
    
    distance_travelled = [0] * len(df_times)

    for i in range(len(df_times)):

        label = Label(root, text= str('Analyzing video ' + str(int(i)) + ' of ' + str(len(df_times))))
        label.pack()

        print('Analyzing video ' + str(int(i)) + ' of ' + str(len(df_times)))

        df = pd.read_csv(os.path.join(csv_direc, df_times['VideoName'][i] + video_suffix), header = [1, 2], index_col = 0)

        int_time_df = df.loc[df_times['Start' + behavior_type + 'Frames'][i]:df_times['Stop' + behavior_type + 'Frames'][i]]

        int_time_df.reset_index(drop = True, inplace = True)
        
        distances = [0] * len(int_time_df)
        count = 0

        for row in range(len(int_time_df)-2):
            count = count + 1
            distances[count] = (dist_formula(int_time_df['nose']['x'].loc[row+1], int_time_df['nose']['x'].loc[row], int_time_df['nose']['y'].loc[row+1], int_time_df['nose']['y'].loc[row]))

        distance_travelled[i] = np.sum(distances)

        print('Just finished video ' + str(int(i)) + ' of ' + str(len(df_times)))

    pd.DataFrame([distance_travelled, df_times['VideoName']]).T.set_index(1).to_csv(os.path.join(main_dir, behavior_type +'_distance_travelled.csv'))

    print('All finished! Distance travelled information is saved in distance_traveled.csv in the main analysis directory')

distance_travelled_btn = Button(tab_frame, text = '8: distance', command = total_distance_travelled).pack(side = 'left')


### finishing the Tkinter loop

root.mainloop()



#### few more things I might want to add ####
## a config file, that is saved automatically with all the directory information for a project, maybe just a csv that we could save each row as a separate path
## heatmap function that comes after the coordinate information is calculated, so that we can plot colored head vectors based on which behavior is happening
## photos or graphics to make it look a bit nicer, although this is definitely not a necessity