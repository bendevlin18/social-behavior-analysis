import os
import subprocess

os.chdir('C:\\Users\\Ben\\Desktop\\labelled_frames\\labelled_frames')
subprocess.call('ffmpeg -framerate 30 -i frame_%01d.png C:\\Users\\Ben\\output.mp4', shell = True)
