###video_making.py########
def export_labelled_frames(df, vname, frame_val, output_dir, investigation = True):
	
	## import all of the necessary packages
	import numpy as np
	import pandas as pd
	import cv2
	import os
	from tqdm import tqdm
	

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
	pbar = tqdm(total=frames)
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
		cv2.imwrite(filename = os.path.join(output_dir,'frame_' + str(count) + '.png'), img = image_new)
		success,image = video.read()
		count += 1
		pbar.update(1)
	pbar.close()

	
###also video_making.py
def ffmpeg_make_video(main_dir, labelled_frames_direc, vname, clear_dir = True):
	import os
	import subprocess
	import sys
	from sys import platform
	
	video_output_dir = os.path.join(main_dir, 'labelled_videos')

	if not os.path.exists(video_output_dir):
		os.mkdir(video_output_dir)

	if sys.platform == 'win32':
		subprocess.call('ffmpeg -framerate 30 -i ' +  labelled_frames_direc + '\\frame_%01d.png ' +  video_output_dir + '\\' + vname.replace(" ", "") + '.mp4', shell = True)
	
	#### CHANGE THIS TO MAC CONVENTION
	elif sys.platform == 'darwin':
		subprocess.call('ffmpeg -framerate 30 -i ' +  labelled_frames_direc + '/frame_%01d.png ' +  video_output_dir + '/' + vname.replace(" ", "") + '.mp4', shell = True)

	### clearing all the images from the hard drive
	pngs = os.listdir(labelled_frames_direc)

	if clear_dir:
		for img in pngs:
			os.remove(os.path.join(labelled_frames_direc,img))
