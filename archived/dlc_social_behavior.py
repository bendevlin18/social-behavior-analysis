

##### FUNCTIONS MASTER #####


def prepare_coordinates(direc, padding = 40):

    """
    Designed to extract the coordinates that were created with the 'define coordinates' function
    in the terminal and import them into appropriate variables
    for jupyter notebook analysis
    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os
    
    ## will want to start things off by loading in the coordinates drawn (this should be done beforehand)

    left_side = np.loadtxt(direc + 'left_side')
    x_chamber = np.loadtxt(direc + 'x_chamber')
    y_chamber = np.loadtxt(direc + 'y_chamber')
    right_side = np.loadtxt(direc + 'right_side')
    middle = np.loadtxt(direc + 'middle')
    
    ## calculate the padded interaction zone, that is 40 pixels larger in every direction (than the base of the chamber)

    x_outer = Polygon(Polygon(x_chamber).buffer(padding).exterior).exterior.coords.xy
    y_outer = Polygon(Polygon(y_chamber).buffer(padding).exterior).exterior.coords.xy
    x_center = np.mean(x_outer[0]), np.mean(x_outer[1])
    y_center = np.mean(y_outer[0]), np.mean(y_outer[1])
    
    x_zone = Polygon(Polygon(x_chamber).buffer(padding).exterior)
    y_zone = Polygon(Polygon(y_chamber).buffer(padding).exterior)
    
    possible_places = {'x_zone': x_zone, 'y_zone': y_zone, 'left_side': left_side, 'middle': middle, 'right_side': right_side}
    extra_coords = {'x_outer': x_outer, 'x_center': x_center, 'y_outer': y_outer, 'y_center': y_center, 'x_chamber': x_chamber, 'y_chamber': y_chamber}
    
    return possible_places, extra_coords



def soc_behavior_heatmap(df, coordinates, videoname, trial_frames = (450, 7500)):

    """
    Plots heatmap of nose locations over the labelled chambers from 'define_coordinates.py'

    Can be useful to visually see if anything is out of place

    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

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
    plt.title(videoname)


def soc_behavior_heatmap_overlap(df, coordinates, video_location, trial_frames = (450,7500)):

    """
    Plots heatmap of nose locations over the labelled chambers from 'define_coordinates.py'

    Can be useful to visually see if anything is out of place

    This function additionally takes in a video_location to plot a still frame of the actual video under the heatmap

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    cap = cv2.VideoCapture(video_location)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((10, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True

    while (fc < 1  and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    frame = buf[0]

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
    ax.imshow(frame)
    plt.xlim(0,frameWidth)
    plt.ylim(0,frameHeight)
    plt.title(video_location[-13:])



def check_orientation(df):
    
    """
    function that finds the distance between nose, and ears, from center of zone.

    takes an entire video dataframe as its input, and iterates through the rows on its own.

    
    returns either 'oriented' or 'not oriented' based on whether the nose is closest to center, in a list that is the length of the dataframe

    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    orientation = ['not_oriented'] * len(df)
    
    z = -1
    
    for index, val in df.iterrows():
        z = z + 1
        
        dist_to_x = np.sqrt(((x_center[0] - df['nose']['x'].loc[index])**2) + ((x_center[1] - df['nose']['y'].loc[index])**2))
        dist_to_y = np.sqrt(((y_center[0] - df['nose']['x'].loc[index])**2) + ((y_center[1] - df['nose']['y'].loc[index])**2))
        
        if dist_to_x > dist_to_y:        
            distance_to_nose = np.sqrt(((y_center[0] - df['nose']['x'].loc[index])**2) + ((y_center[1] - df['nose']['y'].loc[index])**2))
            distance_to_l_ear = np.sqrt(((y_center[0] - df['left ear']['x'].loc[index])**2) + ((y_center[1] - df['left ear']['y'].loc[index])**2))
            distance_to_r_ear = np.sqrt(((y_center[0] - df['right ear']['x'].loc[index])**2) + ((y_center[1] - df['right ear']['y'].loc[index])**2))
        elif dist_to_x < dist_to_y:
            distance_to_nose = np.sqrt(((x_center[0] - df['nose']['x'].loc[index])**2) + ((x_center[1] - df['nose']['y'].loc[index])**2))
            distance_to_l_ear = np.sqrt(((x_center[0] - df['left ear']['x'].loc[index])**2) + ((x_center[1] - df['left ear']['y'].loc[index])**2))
            distance_to_r_ear = np.sqrt(((x_center[0] - df['right ear']['x'].loc[index])**2) + ((x_center[1] - df['right ear']['y'].loc[index])**2))

        ## just look for what distance is the shortest. if that distance is nose, oriented, otherwise not
        
        if distance_to_nose == np.min([distance_to_nose, distance_to_l_ear, distance_to_r_ear]):
            orientation[z] = 'oriented'
            
    return orientation


def check_orientation_single(df, index_loc):

    """

    Same function, just for finding and returning the orientation at one single frame.

    Takes an additional argument - the index location (frame #) in the dataframe you are feeding to it

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    orientation = 'not_oriented'

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


### Should make a csv that contains info for start of each half (compute stop as 300secs after the start, or total_frames = frames/sec * 300)

def prep_time_df(time_df, video_direc):

    """

    Prepares the time_df by calculating end times of both behaviors,

    and determining the start and stop frames based on framerate of each video

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    df_times = time_df
    
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

        cap = cv2.VideoCapture(video_direc + df_times['VideoName'][i] + '.mp4')
        frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
        StartSocialFrames[i] = df_times['StartSocialSec'][i] * frameRate[i]
        StartNovelFrames[i] = df_times['StartNovelSec'][i] * frameRate[i]
        StopSocialFrames[i] = df_times['StopSocialSec'][i] * frameRate[i]
        StopNovelFrames[i] = df_times['StopNovelSec'][i] * frameRate[i]

    df_times['StartSocialFrames'] = StartSocialFrames
    df_times['StartNovelFrames'] = StartNovelFrames
    df_times['StopSocialFrames'] = StopSocialFrames
    df_times['StopNovelFrames'] = StopNovelFrames

    return df_times



def check_coords(coords, possible_places):

    """

    Can take XY coordinates in as a list, and outputs where those coordinates are located (in terms of possible location in the behavior box)

    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os
    
    x = []

    for i in range(len(list(possible_places.values()))):   
        pt = Point(coords)
        if isinstance(list(possible_places.values())[i], Polygon):
            polygon = list(possible_places.values())[i]
        else:
            polygon = Polygon(list(map(tuple, list(possible_places.values())[i])))
        x = np.append(x, polygon.contains(pt))

    return x 



def clean_data(df):

    """
    Still a WIP - want to maybe clean the dataframe frames where the coordinate probability is low

    This brings with it a fair bit of challenges though - so we should wait and see if it is necessary

    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os
    
    for index, value in df.iterrows():
        if df['nose']['likelihood'].loc[index] < 0.3:
            df.drop(index, inplace = True)
    
    return df


def check_climbing(df, coords):

    """
    
    a function that compares nose coordinates to ear coordinates
    
    if they are closer together than some decided threshold - the mouse's head is oriented upward and is climbing
    
    NOT investigating

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    state = ['climbing'] * len(df)
    z = -1
    
    for index, val in df.iterrows():
        z = z + 1
        distance_1 = np.sqrt(((df['left ear']['x'].loc[index] - df['nose']['x'].loc[index])**2) + ((df['left ear']['y'].loc[index] - df['nose']['y'].loc[index])**2))
        distance_2 = np.sqrt(((df['right ear']['x'].loc[index] - df['left ear']['x'].loc[index])**2) + ((df['right ear']['y'].loc[index] - df['left ear']['y'].loc[index])**2))
        distance_3 = np.sqrt(((df['nose']['x'].loc[index] - df['right ear']['x'].loc[index])**2) + ((df['nose']['y'].loc[index] - df['right ear']['y'].loc[index])**2))

        if np.sum(distance_1, distance_2, distance_3) > 5:
            state[z] = 'not_climbing'



def calculate_investigation_times(time_df, csv_direc, behavior_type = 'Social', bodypart = 'nose', video_suffix = 'DLC_resnet50_social_behavior_allMay27shuffle1_250000.csv'):

    """

    BIG master function that can calculate investigation times based on a time dataframe and

    the location of all the CSVs

    It requires that all other functions are either defined, or install as part of the package

    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    possible_locations = [x_chamber, y_chamber, left_side, right_side, middle]
    final_dict = {}
    df_times = time_df

    for i in range(len(df_times)):

        ### first we will want to get the right dataframe, so we should import it based on the df_times location and clean it

        df = pd.read_csv(csv_direc + df_times['VideoName'][i] + video_suffix, header = [1, 2], index_col = 0)
        bodyparts = np.unique(df.columns.get_level_values(0))

        int_df = df.loc[df_times['Start' + behavior_type + 'Frames'][i]:df_times['Stop' + behavior_type + 'Frames'][i]]

        ### need to lose frame information and return it back to 0:end         
        int_df.reset_index(drop=True, inplace = True) 

        arr = np.zeros(shape = (len(int_df), len(bodyparts), len(possible_locations)))

        ### now we should check the coordinates of each bodypart in each frame
        for row in range(len(int_df)):
            for j in range(len(bodyparts)):
                arr[row][j] = check_coord(int_df[bodyparts[j]][['x', 'y']].loc[row].values)

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
        
        print(final_frame_counts)

        final_dict[df_times['VideoName'][i]] = final_frame_counts

        output_df = pd.DataFrame(final_dict, index = ['Somewhere else','X Investigation', 'Y Investigation', 'X Close', 'Y Close']).T
        
        output_df['type'] = [behavior_type] * len(output_df)
        output_df.reset_index(inplace = True)
        output_df.set_index(['index', 'type'], inplace = True)

    return output_df


def convert_to_secs(df_times, output_df, video_direc):

    """
    converts all of the frame counts from 'calculate investigation times' to second counts

    which depends on the frame rate of the video

    """
    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from shapely.geometry import Point
    from shapely.geometry.polygon import Polygon
    import cv2
    import os

    frameRate = np.zeros(shape = len(df_times))
    i = -1
    new_df = pd.DataFrame(columns = output_df.columns, index = output_df.index)
    for index, values in df_times.iterrows():
        i = i + 1
        cap = cv2.VideoCapture(video_direc + values['VideoName'] + '.mp4')
        frameRate[i] = cap.get(cv2.CAP_PROP_FPS)
        index = output_df.index[i]
        new_df.loc[index[0], index[1]] = (output_df.loc[index[0]].loc[index[1]] / frameRate[i]).values
    else:
        pass
        
    return new_df