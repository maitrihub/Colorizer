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

leftimage = cv2.imread('test.jpg')
leftimage = cv2.cvtColor(leftimage, cv2.COLOR_BGR2RGB)
image = cv2.imread('smallbobLeft1.jpg')
imagecolor = cv2.imread('smallbobLeft1.jpg')
clusterimage = cv2.imread('smallbobLeft1.jpg')


#---------LEFT SIDE---------------
#In order to run our program faster, we uploaded a file of the left side from a previous run 
#This way, we had some control over our variables when we were testing, this can be edited to accomodate any left side
def leftsizearrayimage(limage):
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
#Call black and white method
image_arr = blackandwhite(image)
image_greyarr = blackandwhite(clusterimage)
leftimage = leftsizearrayimage(leftimage)

# Change colored images to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imagecolor = cv2.cvtColor(imagecolor, cv2.COLOR_BGR2RGB)
clusterimage = cv2.cvtColor(imagecolor, cv2.COLOR_BGR2RGB)

#get length and width of images in pixels
len_image = len(image)
width_image = len(image[0])


#---------DISTANCE METHOD---------------
def distance(cluster_dict,i,j):
    distance_list = []
    for key,val in cluster_dict.items():
        euc = euclidean(image_arr[val[0]][val[1]], image_arr[i][j])
        distance_list.append((key,euc, (i,j)))
    distance_list = sorted(distance_list, key=lambda x: x[1])
    min_val = distance_list[0]
    return min_val


#---------NEIGHBORS METHOD---------------
#Given a set of coordinates, iterate through the neighbors and append the grey values to a list
def neighbors(coordinate):
    neighborsList = []
#The following method is to get the neighbors and their grey values.
#Seperate x and y coordinates
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



#---------EUCLIDEAN METHOD---------------
# this method is used to determine which color of the 5 randomly selected points is the closet 
#at a given pixel
def euclidean(p1, p2):
    print("this is euc", temp)
    return temp
    

#Given teh five different clusers and the image array
#This method will determine which cluster is associated with the approrirate pixel
#ie, which cluster is the closest distance to the pixel

#creating a kvalue dictionary 
#This holds all the clusters and adds the coordinates of the values that cooresponds to those clusters
kvalue_dict = {}

#---------AVERAGE METHOD---------------
#This method calculates the average value of a list
def Average(lst):
    return (sum(lst) / len(lst))


# cluster_coord for the left side patch
#cluster coord will add the coordinates for coloring ONLY for the left side
cluster_coord = {}
#Right coord holds all the coordinates for the right side of the image 
right_coord = {}
#left coord holds all the coordinates for the left side of the image 
left_coord = {}
#The right side coordinates of GREY image
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
                #ONLY going through the left side
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

            
    for k_key in kvalue_dict.keys():
    # key to search
        red = [element[0] for element in kvalue_dict.get(k_key,[])]
        green = [element[1] for element in kvalue_dict.get(k_key,[])]
        blue = [element[2] for element in kvalue_dict.get(k_key,[])]
        
        red_av = Average(red)
        green_av = Average(green)
        blue_av = Average(blue)
        
        kvalue_dict[k_key] = [red_av, green_av, blue_av]
  
    return min_val

#cluster dict is a global variable that holds the clusters and there corresponding coordinates
cluster_dict = {}
#------------------K MEANS ALGORITHM----------------------------
def kmean_algo():
#
    # k represents the number of clusters 
    # k = 5 for basic 
    # k = 7 for advanced
    k = 5
    x = random.randint(0, len_image-1)
    y = random.randint(0, width_image-1)
    a=-1
    b=-1
    cluster_dict[0] = (x,y)
    # iterating through 1-k to find clustering points
    for i in range(1,k):
        a = random.randint(0, len_image-1)
        b = random.randint(0, width_image-1)
        while ((a==x and b==y) ==True):
            #This is to ensure that we do not have any of the same coordinate values 
            a = random.randint(0, len_image)
            b = random.randint(0, width_image)
        cluster_dict[i] = (a,b)
    #Add the clustering coordinate to the dictionary ^
    min_val = calc_distance(cluster_dict, image_arr) 
    

#------------------COLORING THE LEFT SIDE---------------------------
## color the left side of the picture using clustering
def color_left():
    for key,val in cluster_coord.items():
        for v in val:
            #go through the entire dictionary and change the values of the image pixel accordingly
            image[v[0]][v[1]] = [kvalue_dict[key][0], kvalue_dict[key][1], kvalue_dict[key][2]]

#global variables
#left_Patch will hold ALL coordinates/PATCHES on the left side of the image
left_Patch = []
#coord_patch is an array that holds the coordiates appended
coord_Patch = []
visited = []
neighborList = []
leftpatch_dict = {}    

#------------------CREATING THE PATCHES--------------------------
def leftpatch():  
    #iterate through all possible coordinates on left side
    for i in range(len_image):
        for j in range(int(width_image/2)):
            coord = ((i,j))
            #Find the neighbors of the coordinates to create patches
            neighborList = neighbors(coord)
            #if we have a full set of neighbors – 3x3 patches
            #then append it into the list
            if len(neighborList) == 9:
                #if the conditions are met,append to neighbor list (GREY VALUES)
                left_Patch.append(neighborList)
                #append the coordinates as well
                coord_Patch.append(coord)
                q = tuple(neighborList)
                #adding the neighborlist (PATCH) with the main coordinate to a dictionary 
                leftpatch_dict[q] = coord
            #neighborList = []
                         
    return
                

#------------------EUCLIDAN DISTANACE OF LIST--------------------------
def list_euc(v1,v2):
    dist = math.sqrt(((v1[0]-v2[0])**2) + ((v1[1]-v2[1])**2) + ((v1[2]-v2[2])**2) + ((v1[3]-v2[3])**2) + ((v1[4]-v2[4])**2) + ((v1[5]-v2[5])**2) + ((v1[6]-v2[6])**2) + ((v1[7]-v2[7])**2) + ((v1[8]-v2[8])**2))

    return dist
    
#------------------EUCLIDAN DISTANACE-------------------
def euclidean(v1, v2):
    return sum((p-q)**2 for p, q in zip(v1, v2)) ** .5

#------------------REMOVING DUPLICATES-------------------
def remove_duplicates(inputList):
    #given a list, remove the duplicates in that list
    ret_list = []
    for x in inputList:
        if x not in ret_list:
            ret_list.append(x)
    return ret_list    

#------------------EUCLIDAN DISTANACE OF LIST--------------------------
def euclidDist(a , b):
    dist = np.linalg.norm(a-b)
    return(dist)

#------------------COLORING RIGHT SIDE OF IMAGE---------------
def colored_patch():
    counter = 0
    #get right side of grey image, image_greyarr
    
    #iterate through every coordinate on the right side
    for key in rightgrey_coord.keys(): 
        #find the patch (rightside) for each coordinate
        patch = np.array(neighbors(key))
        
        if len(patch) != 9:
            #The patch will only be considered if it is 3x3
              continue
        
        min_list = []
        top_6 = []
        top_6_dict = {} #This is a dict with eud. as key and left patch as value.
        new_top_6_dict = {}
        
        #removing any duplicates from the series of left patches
        left_new = remove_duplicates(left_Patch)
        #Do the same for coordinates
        coord_new = remove_duplicates(coord_Patch)
        
        neighborList = []
        left_dic = {}
       
        count_lp = 0 
        coord_dict = {}
        #iterate through every patch in the left array 
        for lp in left_new:
            lp = np.array(lp)
            #find the euclidian distance of both patches 
            #append the euclidian distance as a key and teh cooresponding coordinates as the value
            eu = euclidDist(patch,lp)
            #{key,value} = {euclidian distance, coordinates}
            top_6_dict[eu]= coord_new[count_lp]
            count_lp +=1
            
        #sorting the dictionary by EU
        #finding the 6 most similar patches
        for top in sorted(top_6_dict.keys()):
            new_top_6_dict[top] = top_6_dict[top]
            
        medianList = []
        
        #once the list is sorted, we only look at top 6 patches
        for x in list(new_top_6_dict)[0:6]:
            #adding the 6 patches to a list 
            medianList.append((x,new_top_6_dict[x]))
        
        #finding the color of the 6 patches
        color_top623(medianList,key)

        ##1.take each key, find the matching pixel in the right side of the colored gray image, find its 9 neigbhors.
        ## 2. take the 3x3 patch created on the left side and find 6 similar patches on the right side of the gray image. 
        ## 3. take the middle pixel of the six patches found in step 2, and find the matching pixel in the colored clustered
        ## suppose all middle pixels are red, then we know that the key,val in right_coord will be of the color red.


#for each of the top 6 patches, run color_top62
def color_top623(top_6,rightCoord):
    #seperating the 6 patches from the coordinates assoicated with them 
    coord_6_List = []
    for i in range(6):
        coord_6_List.append(top_6[i][1])
    color_top62(coord_6_List,rightCoord)
        
    return    
        
def color_top62(coord_6_List,rightCoord):
    #finding the average of the colors 
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
