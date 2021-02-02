from moviepy.video.io.ffmpeg_tools import *
from moviepy.editor import ImageSequenceClip
from moviepy.editor import VideoFileClip, concatenate_videoclips
import moviepy.editor as mpy
import os
import ffmpeg
import pandas as pd

main_dir = "/Users/Justin/Desktop/crop_practice"

vid_file = "/Users/Justin/Desktop/crop_practice/videos"

file_list = os.listdir(vid_file)

file_list.sort()

print(file_list)



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

#framesFile = "/Users/Justin/Desktop/sample_frames/*.png"
#outputFile = "/Users/Justin/Desktop/frame_movie.mp4"

#stream = ffmpeg.input(framesFile, pattern_type='glob', framerate=30)
#stream = ffmpeg.output(stream,'frame_movie.mp4')
#ffmpeg.run(stream)

#time2 = time.perf_counter()

#print(time2-time1)
 
#write something in the main python to take in a video (From file) and
# allow us to crop it by the start times from df_times and export it to 
# a cropped videos file

#currently using the full video to index csv
#so we need to be able to crop the csv