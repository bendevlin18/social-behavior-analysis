import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector

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
###Annotation.py######## end