#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries 
import os
import pandas as pd
import numpy as np
import sklearn as sk
from matplotlib import pyplot as plt
import cv2
import imutils
import time
from sklearn.cluster import KMeans
from collections import Counter


# In[2]:


path_skin = "Desktop/videos/vid1"


# In[3]:


#defining_skin_shades

rgb_lower = [95,85,36]
rgb_higher = [255,219,172]

#shades_of_different_skin_type
skin_shades = {
    'dark' :   [rgb_lower,[130,116,66]],
    'mulat':   [[175,134,66], [200,172,105]],
    'fair' :   [[200,175,106],[241,194,125]],
    'bright' : [[241,194,125],rgb_higher]
}


# In[4]:


#сreating_dictionary_with_skin_tones
convert_skintones = {}

for shade in skin_shades:
    convert_skintones.update({
        shade : [
            (skin_shades[shade][0][0] * 256 * 256) + (skin_shades[shade][0][1] * 256) + skin_shades[shade][0][2],
            (skin_shades[shade][1][0] * 256 * 256) + (skin_shades[shade][1][1] * 256) + skin_shades[shade][1][2]
        ]
    })


# In[5]:


#extracting_skin_on_the_image
def extractSkin(image):
    img = image.copy()
    #converting_from_bgr_to_hsv
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    #dataframe_with_lower_treshold
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    #dataframe_with_upper_treshold
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    #getting_the_mask
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    #defining_bitwise_and_for_data 
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    #returning_converted_image_from_hsv_to_bgr
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


# In[6]:


#removing_everything_which_is_not_skin_from_the_image
def removeBlack(estimator_labels, estimator_cluster):
    hasBlack = False
    occurance_counter = Counter(estimator_labels)
    def compare(x, y): return Counter(x) == Counter(y)
    for x in occurance_counter.most_common(len(estimator_cluster)):
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]
        if compare(color, [0, 0, 0]) == True:
            del occurance_counter[x[0]]
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break
    return (occurance_counter, estimator_cluster, hasBlack)


# In[7]:


#function_of_getting_color_data
def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):
    occurance_counter = None
    colorInformation = []
    hasBlack = False
    if hasThresholding == True:
        (occurance, cluster, black) = removeBlack(estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black
    else:
        occurance_counter = Counter(estimator_labels)
    totalOccurance = sum(occurance_counter.values())
    for x in occurance_counter.most_common(len(estimator_cluster)):
        index = (int(x[0]))
        index = (index-1) if ((hasThresholding & hasBlack) & (int(index) != 0)) else index
        color = estimator_cluster[index].tolist()
        color_percentage = (x[1]/totalOccurance)
        colorInfo = {"cluster_index": index, "color": color, "color_percentage": color_percentage}
        colorInformation.append(colorInfo)
    return colorInformation


# In[8]:


#extracting_dominant_color
def extractDominantColor(image, number_of_colors = 1, hasThresholding = False):
    if hasThresholding == True:
        number_of_colors += 1
        
    img = image.copy()
    #converting_image_type
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #reshaping
    img = img.reshape((img.shape[0]*img.shape[1]), 3)
    
    #Getting_K-means_model
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)
    #teaching_the_model
    estimator.fit(img)
    
    colorInformation = getColorInformation(estimator.labels_, estimator.cluster_centers_, hasThresholding)
    
    return colorInformation


# In[67]:


#Printing_the_bar_of_dominant_color
def plotColorBar(colorInformation):
    #defining_its_size
    color_bar = np.zeros((100, 500, 3), dtype="uint8")
    
    #initial_state
    top_x = 0
    

    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])
        color = tuple(map(int, (x['color'])))
        cv2.rectangle(color_bar, (int(top_x), 0), (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
        print('Dynamic color range: ', color)
        
    return color_bar


# In[155]:


for i in range(1,11):
    image = imutils.url_to_image("https://github.com/Papa-Vad/ResearchProject/blob/main/vid10/"+str(i)+".jpg?raw=true")
    image = imutils.resize(image, width=250)
    skin = extractSkin(image)
    plt.subplot(3, 1, 2)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
    plt.clf() #comment line if we need image, not only the color
    unprocessed_dominant = extractDominantColor(skin, number_of_colors=1, hasThresholding=True)
    
    decimal_lower = (rgb_lower[0] * 256 * 256) + (rgb_lower[1] * 256) + rgb_lower[2]
    decimal_higher = (rgb_higher[0] * 256 * 256) + (rgb_higher[1] * 256) + rgb_higher[2]
    
    dominantColors = []
    for clr in unprocessed_dominant:
        clr_decimal = int((clr['color'][0] * 256 * 256) + (clr['color'][1] * 256) + clr['color'][2])
        if clr_decimal in range(decimal_lower,decimal_higher+1):
            clr['decimal_color'] = clr_decimal
            dominantColors.append(clr)
    
    skin_tones = []
    if len(dominantColors) == 0:
        skin_tones.append('Unrecognized')
    else:
        for color in dominantColors:
            for shade in convert_skintones:
                if color['decimal_color'] in range(convert_skintones[shade][0],convert_skintones[shade][1]+1):
                    skin_tones.append(shade)
    colour_bar = plotColorBar(dominantColors)
    plt.subplot(3, 1, 3)
    plt.axis("off")
    plt.imshow(colour_bar)
    plt.savefig("Desktop/videos/vid10/color"+str(i)+".jpg",bbox_inches='tight')             


# In[161]:



colorInfo = [{"cluster_index": 0, "color": [152, 127, 110], "color_percentage": 1.0}]
colour_bar = plotColorBar(colorInfo)
plt.subplot(3, 1, 3)
plt.axis("off")
plt.imshow(colour_bar)
#plt.savefig("Desktop/videos/vid6/color9.jpg",bbox_inches='tight')  


# In[180]:


ar = [92,
  88,
 93,
  93,
 92,
 91,
  91,
  92,
  98,
  95]
av = sum(ar)/10
print(av)
av_d=0
for i in range(10):
    av_d+=ar[i]-av
print(av_d/10)    


# In[ ]:




