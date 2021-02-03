
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
	plt.title(region + ' - Press Q to move to the next location')

	## this is the function that is fed to the polygon selector tool. All I am doing here is saving the coordinates for every polygon drawn to a text file of numpy coordinates ##
	## this allows me to save them outside of the python kernel, and they can be pulled in later at any point ##
	def onselect(verts):
		np.savetxt(os.path.join(os.path.join(main_dir, 'coordinates'), region), verts)

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
		plt.title('Example Coordinate Overlay - Press q to quit')
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

def plot_heatmap_dark(coordinates, df, trial_frames):
		fig, ax = plt.subplots(1,1,figsize=(5,3))
		ax.plot(coordinates[0]['left_side'][:,0], coordinates[0]['left_side'][:,1], color = 'red', linewidth = 2)
		ax.plot(coordinates[0]['right_side'][:,0], coordinates[0]['right_side'][:,1], color = 'purple', linewidth = 2)
		ax.plot(coordinates[1]['x_chamber'][:,0], coordinates[1]['x_chamber'][:,1], color = 'blue', linewidth = 2)
		ax.plot(coordinates[1]['y_chamber'][:,0], coordinates[1]['y_chamber'][:,1], color = 'green', linewidth = 2)
		ax.plot(coordinates[1]['x_outer'][0], coordinates[1]['x_outer'][1], color = 'blue', linewidth = 2)
		ax.plot(coordinates[1]['y_outer'][0], coordinates[1]['y_outer'][1], color = 'green', linewidth = 2)
		ax.plot(coordinates[1]['x_center'][0], coordinates[1]['x_center'][1], 'bo')
		ax.plot(coordinates[1]['y_center'][0], coordinates[1]['y_center'][1], 'go')
		ax.hexbin(df['nose']['x'].loc[trial_frames[0]:trial_frames[1]], df['nose']['y'].loc[trial_frames[0]:trial_frames[1]], bins = 10, gridsize = 50, cmap='YlOrRd')

def time_df(df_times, v_location):

	## import the time dataframe and calculate when the sociability and social novelty occur based on start time ##
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
		cap = cv2.VideoCapture(os.path.join(v_location, df_times['VideoName'][i] + '.mp4'))
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


#### function for processing the raw csv files
#### takes likelihood values and smooths out parts where the network is uncertain
def process_csv(df):
    nosex = df['nose']['x'].values
    nosey = df['nose']['y'].values

    leftx = df['left ear']['x'].values
    lefty = df['left ear']['y'].values

    rightx = df['right ear']['x'].values
    righty = df['right ear']['y'].values

    tailx = df['tail']['x'].values
    taily = df['tail']['y'].values

    for i in range(len(df) - 1):
        i = i+1
        if df['nose']['likelihood'].loc[i] < 0.98:
            nosex[i] = nosex[i-1]
            nosey[i] = nosey[i-1]

            rightx[i] = rightx[i-1]
            righty[i] = righty[i-1]

            leftx[i] = leftx[i-1]
            lefty[i] = lefty[i-1]

            tailx[i] = tailx[i-1]
            taily[i] = taily[i-1]
            
    df_processed = pd.DataFrame([nosex, nosey, rightx, righty, leftx, lefty, tailx, taily]).T
    colnames = df.T.loc[[('nose', 'x'), ('nose', 'y'), ('right ear', 'x'), ('right ear', 'y'), ('left ear', 'x'), ('left ear', 'y'), ('tail', 'x'), ('tail', 'y')]].T.columns
    
    df_processed.columns = colnames
    
    return df_processed


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

### export labelled frames function that allows you to take a video and create labelled frames with the output head vector
### REQUIRES VIRTUALDUB TO STITCH THE VIDEO TOGETHER AFTER ###

def export_labelled_frames(df, vname, frame_val, output_dir = 'labelled_frames', investigation = True):
	
	## import all of the necessary packages
	import numpy as np
	import pandas as pd
	import cv2
	import os
	
	video = cv2.VideoCapture(vname)

	print('Starting to save labelled video frames')

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	## extract relevant meta information about the video
	frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(video.get(cv2.CAP_PROP_FPS))
	
	## while loop through every frame of the video and label each frame
	success, image = video.read()
	count = 0
	while success:
		print(count / len(df))
		nose_coords = (int(df['nose']['x'].loc[count]), int(df['nose']['y'].loc[count]))
		midpoint_coords = (int((df['right ear']['x'].loc[count] + df['left ear']['x'].loc[count]) / 2) , int((df['right ear']['y'].loc[count] + df['left ear']['y'].loc[count]) / 2))
		print(nose_coords, midpoint_coords)
		if frame_val[count] == 'Somewhere else':
			color = (0, 0, 255)
		if frame_val[count] == 'X Close':
			color = (0, 0, 255)
		if frame_val[count] == 'Y Close':
			color = (0, 0, 255)
		if frame_val[count] == 'X Investigation':
			color = (0, 255, 0)
		if frame_val[count] == 'Y Investigation':
			color = (0, 255, 0)
		image_new = cv2.line(image, nose_coords, midpoint_coords, color, 4)
		cv2.imwrite(filename = os.path.join(output_dir, 'frame_' + str(count) + '.png'), img = image_new)
		success,image = video.read()
		count += 1



### distance formula for calculating the total distance travelled for each animal

def dist_formula(x1, y1, x2, y2):
	d = np.sqrt((x2 + x1)**2 + (y2 - y1)**2)

	return d



### funtion for calculation investigation times on only one video

def calculate_investigation_times_single(df, possible_places, extra_coords):
	
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	from shapely.geometry import Point
	from shapely.geometry.polygon import Polygon
	import cv2
	
	bodyparts = np.unique(df.columns.get_level_values(0))[1:]

	arr = np.zeros(shape = (len(df), len(bodyparts), len(possible_places)))

	### now we should check the coordinates of each bodypart in each frame
	print('Calculating Investigation Times: ')
	for row in range(len(df)):
		print(row / len(df))
		for j in range(len(bodyparts)):
			arr[row][j] = check_coords(df[bodyparts[j]][['x', 'y']].loc[row].values, possible_places)
			
	print('Array Constructed!')

	### set which patterns mean x vs y investigation, only for the first three bodyparts (nose and ears, cuz we don't care about tail base yet)
	x_inv = np.array([[1., 0., 1., 0., 0.]])
	y_inv = np.array([[0., 1., 0., 0., 1.]])

	### now we want to check each frame in our array, and create a frame_val array that holds info about where the mouse's head was detected
	z = -1
	frame_val = np.zeros(shape = len(arr), dtype = 'object')
	for frame in range(len(arr)):
		print(int(100*(frame / len(arr))))
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








	##### WORK IN PROGRESS #####


def label_frames(df, vname, frame_val, output_dir = 'labelled_frames', investigation = True):
	
## import all of the necessary packages
	import numpy as np
	import pandas as pd
	import cv2
	import os
	
	video = cv2.VideoCapture(vname)

	print('Starting to save labelled video frames')

	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	
	## extract relevant meta information about the video
	frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(video.get(cv2.CAP_PROP_FPS))
	full_array = []
	## while loop through every frame of the video and label each frame
	success, image = video.read()
	count = 0
	while success:
		nose_coords = (int(df['nose']['x'].loc[count]), int(df['nose']['y'].loc[count]))
		midpoint_coords = (int((df['right ear']['x'].loc[count] + df['left ear']['x'].loc[count]) / 2) , int((df['right ear']['y'].loc[count] + df['left ear']['y'].loc[count]) / 2))
		if frame_val[count] == 'Somewhere else':
			color = (0, 0, 255)
		if frame_val[count] == 'X Close':
			color = (0, 0, 255)
		if frame_val[count] == 'Y Close':
			color = (0, 0, 255)
		if frame_val[count] == 'X Investigation':
			color = (0, 255, 0)
		if frame_val[count] == 'Y Investigation':
			color = (0, 255, 0)
		image_new = cv2.line(image, nose_coords, midpoint_coords, color, 4)
		full_array.append(image_new) 
		#cv2.imwrite(filename = os.path.join(output_dir, 'frame_' + str(count) + '.png'), img = image_new)
		success,image = video.read()
		count += 1
		print(count)

	out = cv2.VideoWriter(os.path.join(output_dir, 'project_testing.avi'),cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

	for i in range(len(full_array)):
		out.write(full_array[i])
	out.release()