from moviepy.video.io.ffmpeg_tools import *
from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.editor as mpy
import os
import ffmpeg
import pandas as pd
from tkinter import filedialog
from analysis_functions_master import *

main_dir = "Users/Justin/Desktop/crop_practice"
frame_values_folder = os.path.join(main_dir, 'frame_values')

#possible_places and extra_coords are all defined earlier and are not dependent on the video

#defines df as csv for video of interest
#could we just change this to the frame_val csv?
csv_to_label = filedialog.askopenfilename(initialdir = frame_values_folder, title = 'Select the corresponding csv file')
csv_direc = os.path.join(main_dir, frame_values_folder)
df = pd.read_csv(os.path.join(csv_direc, csv_to_label), header = [0, 1]).dropna().reset_index(drop = True)
print(os.path.join(csv_direc, csv_to_label))
print(df)

def calculate_investigation_times_single(df, possible_places, extra_coords):
	
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	import cv2
	from tqdm import tqdm
	
	frame_val = df

	### now we want to check each frame in our array, and create a frame_val array that holds info about where the mouse's head was detected
	z = -1
	frame_val = np.zeros(shape = len(arr), dtype = 'object')
	for frame in tqdm(range(len(arr))):
		z = z + 1
		comparison_x = arr[frame][0:1] == x_inv
		comparison_y = arr[frame][0:1] == y_inv

		if comparison_x.all() == True:
			if check_orientation_single(df, z, extra_coords) == 'oriented':
				frame_val[z] = 'X Investigation'
			elif check_orientation_single(df, z, extra_coords) == 'not_oriented':
				frame_val[z] = 'X Close'
		elif comparison_y.all() == True:
			if check_orientation_single(df, z, extra_coords) == 'oriented':
				frame_val[z] = 'Y Investigation'
			elif check_orientation_single(df, z, extra_coords) == 'not_oriented':
				frame_val[z] = 'Y Close'
		else:
			frame_val[z] = 'Somewhere else'
		
	print('Investigation Times Calculated!!')

	return frame_val

# main_dir = "/Users/Justin/Desktop/crop_practice"

# vid_file = "/Users/Justin/Desktop/crop_practice/videos"

# file_list = os.listdir(vid_file)

# file_list.sort()

# print(file_list)



#video2 = os.getcwd()

#video1 = "/Users/Justin/Desktop/videos/AC_SOC2_1.mp4"
#start_time = 22
#end_time = start_time + 20
#start2 = 406
#end2 = start2 + 20

#output1 = "/Users/Justin/Desktop/videos/AC_SOC2_1_cropped_1.mp4"
#output2 = "/Users/Justin/Desktop/videos/AC_SOC2_1_cropped_2.mp4"

#ffmpeg_extract_subclip(video1, start_time,end_time, targetname=output1)
#ffmpeg_extract_subclip(video1, start2,end2, targetname=output2)

#output3 = "/Users/Justin/Desktop/videos/AC_SOC2_1_cropped_3.mp4"




#cropFile = "/Users/Justin/Desktop/videos/crops"

#stream = ffmpeg.input(output1)
#stream2 = ffmpeg.input(output2)
#stream = ffmpeg.concat(stream, stream2)
#stream = ffmpeg.output(stream, 'output_concat.mp4')
#print("before run")
#ffmpeg.run(stream)
#print("after run")
#import time
#time1 = time.perf_counter()

# framesFile = "/Users/Justin/Desktop/sample_frames/*.png"
# outputFile = "/Users/Justin/Desktop/frame_movie.mp4"

# #stream = ffmpeg.input(framesFile, pattern_type='glob', framerate=30)
# #stream = ffmpeg.output(stream,'frame_movie.mp4')
# #ffmpeg.run(stream)

# csv_direc = "/Users/Justin/Desktop/crop_practice/smoothed_csv_output/"

# first_vid = os.listdir(csv_direc)[0]
# print(first_vid)
# video_suffix_start = first_vid.index("DLC")
# video_suffix = first_vid[video_suffix_start: len(first_vid)]

# from tkinter import *
# from tkinter import simpledialog
# root = Tk()
# root.grid_rowconfigure(0, weight=1)
# root.grid_columnconfigure(0, weight=1)
# #behavior_type = simpledialog.Dialog(root, title="Which type of behavior?")
# direc_frame = LabelFrame(root, padx = 5, pady = 5)
# direc_frame.grid(padx = 10, pady = 10, sticky='nsew')
# df_times_frame = LabelFrame(root, padx = 5, pady = 5)
# df_times_frame.grid(padx = 10, pady = 10, sticky='nsew')
# tab_frame = LabelFrame(root, padx = 5, pady = 5)
# tab_frame.grid(padx = 10, pady = 10, sticky='nsew')

# main_direc_label = Label(direc_frame, text = 'Which type of behavior?').grid(row = 1, column = 0, sticky='nsew')
# main_direc_button = Button(tab_frame, text = 'Social', command = create_coord_window).pack(side = 'left')
# main_coord_import_btn = Button(tab_frame, text = 'Novel', command = create_coord_window).pack(side = 'right')

#time2 = time.perf_counter()

#print(time2-time1)
 
#write something in the main python to take in a video (From file) and
# allow us to crop it by the start times from df_times and export it to 
# a cropped videos file

#currently using the full video to index csv
#so we need to be able to crop the csv