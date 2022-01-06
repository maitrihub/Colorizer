
#---------IMPORT STATEMENTS----------------
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import sys
import math
import statistics
import webcolors
from PIL import ImageColor
sys.maxsize
np.seterr(over='ignore')

leftimage = cv2.imread('half3.png')
leftimage = cv2.cvtColor(leftimage, cv2.COLOR_BGR2RGB)
image = cv2.imread('smallbob.png')
imagecolor = cv2.imread('smallbob.png')
clusterimage = cv2.imread('smallbob.png')
learning_rate = .5

#---------LEFT SIDE---------------
#In order to run our program faster, we uploaded a file of the left side from a previous run 
#This way, we had some control over our variables when we were testing, this can be edited to accomodate any left side

def leftsizearrayimage(limage):
#Iterating through every pixel in the code and converting it to greyscale
    for x in range(len(limage)):
        for y in range(len(limage[x])):
            red = limage[x][y][0] 
            green = limage[x][y][1]
            blue= limage[x][y][2]
            limage[x][y]  = [limage[x][y][0] ,limage[x][y][1] ,limage[x][y][2]]
    return limage



#---------black and white---------------
#Converting the image to black and white, pixel by pixel 
def blackandwhite(image):
#Iterating through every pixel in the code and converting it to greyscale
    for x in range(len(image)):
        for y in range(len(image[x])):
            image[x][y] = 0.21*image[x][y][0] + 0.72*image[x][y][1] + 0.07*image[x][y][2]
    return image
# Read in the image

image_arr = blackandwhite(image)
image_greyarr = blackandwhite(clusterimage)
leftimage = leftsizearrayimage(leftimage)

# Change color to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imagecolor = cv2.cvtColor(imagecolor, cv2.COLOR_BGR2RGB)
clusterimage = cv2.cvtColor(imagecolor, cv2.COLOR_BGR2RGB)

len_image = len(image)
width_image = len(image[0])
#print(width_image)
#print(len_image)

#---------NEIGHBORS METHOD---------------
def neighbors(coordinate):
    neighborsList = []
#The following method is to get the neighbors and their grey values.
    x = coordinate[0]
    y = coordinate[1]
    x = int(x)
    y = int(y)
    neighborsList.append(image_arr[x][y][0])
    if(x+1 < width_image):
        neighborsList.append(image_arr[x+1][y][0])
    if(x-1 >= 0):  
        neighborsList.append(image_arr[x-1][y][0])
    if(y+1 < len_image):  
        neighborsList.append(image_arr[x][y+1][0])
    if(y-1 >= 0): 
        neighborsList.append(image_arr[x][y-1][0])
    if (x+1 < width_image and y+1< len_image):
        neighborsList.append(image_arr[x+1][y+1][0])
    if (x+1 < width_image and y-1 >= 0):   
        neighborsList.append(image_arr[x+1][y-1][0])    
    if (x-1 >=0 and y+1< len_image):   
        neighborsList.append(image_arr[x-1][y+1][0])
    if (x-1>=0 and y-1 >=0):  
        neighborsList.append(image_arr[x-1][y-1][0])
        
    return neighborsList


#The distance method calls teh euclidean method to 
#append it into a distance list 
#From here, the minimum value of the eucldian distance is returned
def distance(cluster_dict,i,j):
    distance_list = []
    for key,val in cluster_dict.items():
        euc = euclidean(image_arr[val[0]][val[1]], image_arr[i][j])
        distance_list.append((key,euc, (i,j)))
    distance_list = sorted(distance_list, key=lambda x: x[1])
    min_val = distance_list[0]
    return min_val

#Given the five different clusers and the image array
#This method will determine which cluster is associated with the approrirate pixel
#ie, which cluster is the closest distance to the pixel
kvalue_dict = {}
def Average(lst):
    return (sum(lst) / len(lst))
## cluster_coord for the left side patch
cluster_coord = {}
right_coord = {}
left_coord = {}
rightgrey_coord = {}


#---------CALCULATE DISTANCE METHOD--------------
def calc_distance(cluster_dict, image_arr):
    length = sum(len(row) for row in image_arr)
    distance_dict = {}
#appending each left side coordinate to distance_dict
    for key in cluster_dict.keys():
        distance_dict[key] = []
        
    #iterate through the length and width of the image 
    #which corresponds to (i,j) values
    for i in range(len_image):
        for j in range(width_image):
            #find the minimum distance
            min_val = distance(cluster_dict,i,j)
            half_width = float(width_image/2)
            half_len = float(len_image/2)
            if min_val[0] not in kvalue_dict.keys():
                x_coord = min_val[2][0]
                y_coord = min_val[2][1]
                kvalue_dict[min_val[0]] = [imagecolor[x_coord][y_coord]]
                if (float(y_coord) < (width_image/2)):
                    cluster_coord[min_val[0]] = [(x_coord, y_coord)]
                else:
                    right_coord[(x_coord, y_coord)] = [min_val[0]]
                
                
            else:
                ## for the left half
                x_coord = min_val[2][0]
                y_coord = min_val[2][1]
                kvalue_dict[min_val[0]].append(imagecolor[x_coord][y_coord])
                if min_val[0] not in cluster_coord.keys():
                    if (float(y_coord) < (width_image/2)):
                        cluster_coord[min_val[0]] = [(x_coord, y_coord)]
                else:
                    if (float(y_coord) < half_width):
                        cluster_coord[min_val[0]].append((x_coord, y_coord))
                
               ### for the right half 
                if min_val[0] not in right_coord.keys():
                    if (float(y_coord) > (width_image/2)):
                        right_coord[(x_coord, y_coord)] = [min_val[0]]
                        rightgrey_coord[(x_coord, y_coord)] = [min_val[0]]
                else:
                    if (float(y_coord) > half_width):
                        right_coord[(x_coord, y_coord)].append(min_val[0])
                        rightgrey_coord[(x_coord, y_coord)].append(min_val[0])

            #print(min_val)  
    for k_key in kvalue_dict.keys():
    # key to search
        red = [element[0] for element in kvalue_dict.get(k_key,[])]
        green = [element[1] for element in kvalue_dict.get(k_key,[])]
        blue = [element[2] for element in kvalue_dict.get(k_key,[])]
        
        red_av = Average(red)
        green_av = Average(green)
        blue_av = Average(blue)
        
        kvalue_dict[k_key] = [red_av, green_av, blue_av]

    #print(right_coord)       
    return min_val

#cluster dict is a global variable that holds the clusters and there corresponding coordinates
cluster_dict = {}
#------------------K MEANS ALGORITHM----------------------------
def kmean_algo():
    
    k=7
    x = random.randint(0, len_image-1)
    y = random.randint(0, width_image-1)
    a=-1
    b=-1
    cluster_dict[0] = (x,y)
    for i in range(1,k):
        a = random.randint(0, len_image-1)
        b = random.randint(0, width_image-1)
        while ((a==x and b==y) ==True):
            #print("here")
            a = random.randint(0, len_image)
            b = random.randint(0, width_image)
        cluster_dict[i] = (a,b)
    #print(cluster_dict)
    min_val = calc_distance(cluster_dict, image_arr) 

## color the left side of the picture using clustering
def color_left():
    for key,val in cluster_coord.items():
        for v in val:
            #print(kvalue_dict[key][0], kvalue_dict[key][1], kvalue_dict[key][2])
            image[v[0]][v[1]] = [kvalue_dict[key][0], kvalue_dict[key][1], kvalue_dict[key][2]]


left_Patch = []
coord_Patch = []
visited = []
neighborList = []
#------------------COLORING THE LEFT SIDE---------------------------    
## color the left side of the picture using clustering
leftpatch_dict = {}    
def leftpatch():  
    for i in range(len_image):
        for j in range(int(width_image/2)):
            coord = ((i,j))
            neighborList = neighbors(coord)
            #print(coord,neighborList,'\n')
            if len(neighborList) == 9:
                left_Patch.append(neighborList)
                coord_Patch.append(coord)
                q = tuple(neighborList)
                #leftpatch_dict[neighborList] = coord
                leftpatch_dict[q] = coord
             
    return
                
      
       
#removes the dupicate values from a list and returns a list with distinct values
def remove_duplicates(inputList):
    ret_list = []
    for x in inputList:
        if x not in ret_list:
            ret_list.append(x)
    return ret_list    


#weightage calculated based on the distance
# used as an additional function that can get more accurate results for our k means regression
def weighted (dist):
    if dist != 0:
        weight = (1/(dist**2))
        weightdist = weight * dist
        #print(weight)
        return weightdist
    else:
        return 0
#------------------EUCLIDAN DISTANACE-------------------
def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    dist = weighted(dist)
    return dist
#------------------COLORING RIGHT SIDE OF IMAGE---------------
def colored_patch():
    print('inside of colored patch')
    ##checks the gray side in order color. i.e training colored image
    counter = 0
    #get right side of grey image, image_greyarr
    for key in rightgrey_coord.keys(): 
        ## 3x3 patch of the grayscale values
        patch = np.array(neighbors(key))
        
        #we only consider patches that are of size of 9, and ignore the edge pixels 
        #because their neighbors would be 6
        if len(patch) != 9:
            continue
        min_list = []
        top_6 = []
        top_6_dict = {} #This is a dict with eud. as key and left patch as value.
        new_top_6_dict = {}
        
        
        left_new = remove_duplicates(left_Patch)
        coord_new = remove_duplicates(coord_Patch)
        neighborList = []
        left_dic = {}

        count_lp = 0 
        coord_dict = {}
        #iterate through all the patches and store them using weighted euclidean distance
        for lp in left_new:
            lp = np.array(lp)
            eu = euclidDist(patch,lp)
            eu = weighted(eu)
            top_6_dict[eu]= coord_new[count_lp]
            #print(coord_new[count_lp])
            count_lp +=1
            
        #sort the top_6 dict in terms of their euclidean distance.
        for top in sorted(top_6_dict.keys()):
            new_top_6_dict[top] = top_6_dict[top]
            
        medianList = []
        

        #we use the top 10 left patches, to have more data when calculatiing the average.
        for x in list(new_top_6_dict)[0:10]:
            medianList.append((x,new_top_6_dict[x]))
        #print(medianList)
        
        color_top623(medianList,key)
            
        ##1.take each key, find the matching pixel in the right side of the colored gray image, find its 9 neigbhors.
        ## 2. take the 3x3 patch created on the left side and find 6 similar patches on the right side of the gray image. 
        ## 3. take the middle pixel of the six patches found in step 2, and find the matching pixel in the colored clustered
        ## we created. suppose all middle pixels are red, then we know that the key,val in right_coord will be of the color red.

        
#for each of the top 6 patches, run color_top62
def color_top623(top_6,rightCoord):
    #seperating the 6 patches from the coordinates assoicated with them 
    coord_6_List = []
    for i in range(10):
        coord_6_List.append(top_6[i][1])
    color_top62(coord_6_List,rightCoord)
    
        
    return    

#Finding the average for thee RGB values
def color_top62(coord_6_List,rightCoord):
    redArray = []
    greenArray = []
    blueArray = []
    for coord in coord_6_List:
        redArray.append(leftimage[coord][0])
        greenArray.append(leftimage[coord][1])
        blueArray.append(leftimage[coord][2])
    #adding the average to the image and coloring, pixel by pixel
    image[rightCoord] = (Average(redArray),Average(greenArray),Average(blueArray))
    
    
kmean_algo()
leftpatch()
color_left()
print('probs outside of left patch')
colored_patch()
#plt.imshow(imagecolor)
plt.imshow(image)
