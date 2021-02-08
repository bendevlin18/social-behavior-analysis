###HeatMap.py#######
#Want to figure out a better way to plot the heatmaps
#want a better color map; more options, want to handle start times being off
#play around with bins, hexsize, colormap
def plot_heatmap(coordinates, df, trial_frames):

    import matplotlib.pyplot as plt
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

    import matplotlib.pyplot as plt

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
###HeatMap.py####### end