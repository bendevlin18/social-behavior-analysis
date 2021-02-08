###Utils.py########
#written generalizable
##### Functions for main calculations #####

def check_coords(coords, possible_places):

    import numpy as np
    from shapely.geometry import Point
    from shapely.geometry import Polygon

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

###Utils.py
#comment that it still hasn't been implemented
def check_climbing(df, coords):
    import numpy as np

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

###utils.py
### distance formula for calculating the total distance travelled for each animal

def dist_formula(x1, y1, x2, y2):
    import numpy as np

    d = np.sqrt((x2 + x1)**2 + (y2 - y1)**2)

    return d
