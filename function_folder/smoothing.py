###smoothing.py########
### extracting index locations for any uncertain groups
def extract_uncertain_groups(df, bodypart = 'nose', uncertainty = 0.95):
	import numpy as np
	import pandas as pd
	import more_itertools as mit
	import os
	
	iterable = df[df[bodypart]['likelihood'] < uncertainty].index.values
    
	return [list(group) for group in mit.consecutive_groups(iterable)]


### building a function to increment the averages between the two values
def increment_average(two_vals, inc_len):

    import numpy as np
    
    increment = (two_vals[1] - two_vals[0]) / (inc_len + 2)
    vals = []
    temp = two_vals[0]
    for i in range(inc_len):
        temp = temp + increment
        vals = np.append(vals, temp)
        
    return vals


#### function for processing the raw csv files
#### takes likelihood values and smooths out parts where the network is uncertain

#add additional argument to set the uncertainty to be set whatever you want
def process_csv(df):

	import pandas as pd
	
	### need to define a list of all the bodyparts and the two coordinates
	bodyparts = ['nose', 'right ear', 'left ear', 'tail']
	coordinates = ['x', 'y']

	### loop through the bodyparts
	for bodypart in bodyparts:
		### step 1, extract uncertainties 
		uncertain_ilocs = extract_uncertain_groups(df, uncertainty = 0.95, bodypart = bodypart)
		### step 2, loop through x and y coordinates, assign incrementally averaged values
		for coord in coordinates:
			print('Smoothing out: ', bodypart + ' ' + coord)
			for group in uncertain_ilocs:
				try:
					df.replace(df[bodypart][coord][group].values, increment_average([df[bodypart][coord][group[0] - 1], df[bodypart][coord][group[-1] + 1]], len(group)), inplace = True)
				except:
					pass

	df.set_index(pd.RangeIndex(stop = len(df)), inplace = True)

	return df

### Smoothing.py ######### end