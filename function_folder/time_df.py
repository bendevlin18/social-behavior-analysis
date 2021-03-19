###Time_df.py########
#Might want to prompt user about experiment type before time_df
#^will error out now if only Social, since it needs novel start times
#can make 3 separate functions for both, soc only or novel only
def time_df(df_times, v_location, experiment_type):

    import numpy as np
    import cv2
    import os


	## import the time dataframe and calculate when the sociability and social novelty occur based on start time ##
	## in the future, it would be nice to be able to specify ONLY soc, or ONLY snp, or both ##
    df_times.dropna(inplace = True)

    if experiment_type == 'Social':
        df_times['StopSocialSec'] = df_times['StartSocialSec'] + 300

        ### we want to iterate through the rows, grab the video name, open corresponding video, extract framerate, and then multiply startSec cols

        frameRate = np.zeros(len(df_times))
        StartSocialFrames = np.zeros(len(df_times))
        StopSocialFrames = np.zeros(len(df_times))

        for i in range(len(df_times)):
            cap = cv2.VideoCapture(os.path.join(v_location, df_times['VideoName'][i] + '.mp4'))
            frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
            StartSocialFrames[i] = df_times['StartSocialSec'][i] * frameRate[i]
            StopSocialFrames[i] = df_times['StopSocialSec'][i] * frameRate[i]



        df_times['StartSocialFrames'] = StartSocialFrames
        df_times['StopSocialFrames'] = StopSocialFrames

    if experiment_type == 'Novel':
        df_times['StopNovelSec'] = df_times['StartNovelSec'] + 300

        ### we want to iterate through the rows, grab the video name, open corresponding video, extract framerate, and then multiply startSec cols

        frameRate = np.zeros(len(df_times))
        StartNovelFrames = np.zeros(len(df_times))
        StopNovelFrames = np.zeros(len(df_times))

        for i in range(len(df_times)):
            cap = cv2.VideoCapture(os.path.join(v_location, df_times['VideoName'][i] + '.mp4'))
            frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
            StartNovelFrames[i] = df_times['StartNovelSec'][i] * frameRate[i]
            StopNovelFrames[i] = df_times['StopNovelSec'][i] * frameRate[i]

        df_times['StartNovelFrames'] = StartNovelFrames
        df_times['StopNovelFrames'] = StopNovelFrames

    if experiment_type == 'Both':

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

###Time_df.py######## end