# -*- coding: UTF-8 -*-
import os
import glob
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load file path and name
file_path = glob.glob('PATH_TO_YOUR_DATASET/*.png')

for f in file_path:
    img = cv2.imread(f)
    title, ext = os.path.splitext(f)
    img_r = cv2.resize(img, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    print(title)
    dst = cv2.medianBlur(img, ksize=5)
    img_rf = cv2.resize(dst, dsize=(256, 256), interpolation = cv2.INTER_NEAREST)
    dst_af = cv2.medianBlur(img_rf, ksize=3)

    img_Lab = cv2.cvtColor(dst_af, cv2.COLOR_BGR2Lab)
    img_L, img_a, img_b = cv2.split(img_Lab)
    img_array = np.asarray(img_Lab)
    img_array[:,:,0] = img_L * 0.1
    (h,w,c) = img_array.shape
    data_points = img_array.reshape(h*w, c)

    # Initial setting for K-Means clustering
    kmeans_model = KMeans(n_clusters=25) # Specify the number of clusters
    cluster_labels = kmeans_model.fit_predict(data_points)

    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)

    for k in range(25):
        if(rgb_cols[k][0]<2): #Black
            rgb_cols[k][0] = 0
            rgb_cols[k][1] = 127
            rgb_cols[k][2] = 127

        elif( 2<= rgb_cols[k][0] < 8): # Low luminance 
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  100): #Black
                rgb_cols[k][0] = 0
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                
            elif((rgb_cols[k][1] - 127) * (-0.2) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.7 + 127 > rgb_cols[k][2]): #Carmine
                rgb_cols[k][0] = 7
                rgb_cols[k][1] = 170
                rgb_cols[k][2] = 150
               
            elif((rgb_cols[k][1] - 127) * 0.7 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-4.8) + 127 < rgb_cols[k][2]): #Brown
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 145
                rgb_cols[k][2] = 157

            elif((rgb_cols[k][1] - 127) * (-4.8) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3.8) + 127 < rgb_cols[k][2]): #Matcha (powdered ceremonial tea)
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 110
                rgb_cols[k][2] = 160

            elif((rgb_cols[k][1] - 127) * (-3.8) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.3 + 127 < rgb_cols[k][2]): #Green
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
 
            elif((rgb_cols[k][1] - 127) * 1.3 + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1) + 127 > rgb_cols[k][2]): #Indigo
                rgb_cols[k][0] = 7
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 70

            elif((rgb_cols[k][1] - 127) * (-1) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.2) + 127 > rgb_cols[k][2]): #Purple
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95

        elif(8<= rgb_cols[k][0] <12): # Middle luminance
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) > 50 and (rgb_cols[k][1] - 127) * (-3.5) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 127 < rgb_cols[k][2]): #Matcha (powdered ceremonial tea)
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 110
                rgb_cols[k][2] = 160
               
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) > 50 and (rgb_cols[k][1] - 127) * (-2) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.7 + 127 < rgb_cols[k][2]): #Green
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
              
            elif(((rgb_cols[k][1] - 127)*2) + (((rgb_cols[k][2] - 127)**2)) <  226): #Gray
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  1550 and (rgb_cols[k][1] - 127) * 0.2 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.2 + 127 > rgb_cols[k][2]): #Hada (Skin)
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  1550 and (rgb_cols[k][1] - 127) * (-0.2) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.2 + 127 > rgb_cols[k][2]): #Pink
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                
            elif((rgb_cols[k][1] - 127) * (-0.2) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.2 + 127 > rgb_cols[k][2]): #Red
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 200
                rgb_cols[k][2] = 180
                
            elif((rgb_cols[k][1] - 127) * (1.2) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3.5 + 127 > rgb_cols[k][2]): #Brown
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 145
                rgb_cols[k][2] = 157
                
            elif((rgb_cols[k][1] - 127) * 3.5 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3.5) + 127 < rgb_cols[k][2]): #Oudo (sand/mud)
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 170
                
            elif((rgb_cols[k][1] - 127) * 0.7 + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.7) + 127 > rgb_cols[k][2]): #Blue
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 118
                rgb_cols[k][2] = 50
              
            elif((rgb_cols[k][1] - 127) * (-1.7) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.2) + 127 > rgb_cols[k][2]): #Purple
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95

        elif( 12<= rgb_cols[k][0] <15 ): # Middle luminance
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) > 50 and (rgb_cols[k][1] - 127) * (-4) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2.5) + 127 < rgb_cols[k][2]): #Matcha (powdered ceremonial tea)
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 110
                rgb_cols[k][2] = 160
              
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) > 50 and(rgb_cols[k][1] - 127) * (-2.5) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 127 < rgb_cols[k][2]): #Green
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) < 226): #Gray
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  1550 and (rgb_cols[k][1] - 127) * 3.7 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-4) + 127 < rgb_cols[k][2] ): #Yellow
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 150
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  1550 and (rgb_cols[k][1] - 127) * 0.3 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.9 + 127 > rgb_cols[k][2]): #Hada (Skin)
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  4800 and (rgb_cols[k][1] - 127) * (-0.3) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (0.3) + 127 > rgb_cols[k][2]): #Hada (Skin)
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) >= 4800  and (rgb_cols[k][1] - 127) * 4 + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.8) + 127 > rgb_cols[k][2]): #Mizu (water)
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 118
                rgb_cols[k][2] = 50
                
            elif((rgb_cols[k][1] - 127) * 3.7 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-4) + 127 < rgb_cols[k][2]): #Oudo (sand/mud)
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 170
                
            elif((rgb_cols[k][1] - 127) * 0.35 + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.8) + 127 > rgb_cols[k][2]): #Mizu (water)
                rgb_cols[k][0] = 20
                rgb_cols[k][1] = 112
                rgb_cols[k][2] = 103
                
            elif((rgb_cols[k][1] - 127) * (-1.8) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.3) + 127 > rgb_cols[k][2]): #Purple
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                
            elif((rgb_cols[k][1] - 127) * (-0.3) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.9 + 127 > rgb_cols[k][2]): #Red
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 200
                rgb_cols[k][2] = 180
                
            elif((rgb_cols[k][1] - 127) * 0.9 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.9 + 127 > rgb_cols[k][2]): #Orange
                rgb_cols[k][0] = 18
                rgb_cols[k][1] = 180
                rgb_cols[k][2] = 200
                
            elif((rgb_cols[k][1] - 127) * 1.9 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3.7 + 127 > rgb_cols[k][2]): #Brown
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 145
                rgb_cols[k][2] = 157
                
        elif( 15<= rgb_cols[k][0] <21 ): # Middle luminance
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) > 50 and (rgb_cols[k][1] - 127) * (-3) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 127 < rgb_cols[k][2]): #Green
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  82): #White
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  1550 and (rgb_cols[k][1] - 127) * 3 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3) + 127 < rgb_cols[k][2]): #Cream
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 150
                
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) <  1550 and (rgb_cols[k][1] - 127) * 0.6 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3 + 127 > rgb_cols[k][2]): #Hada (skin)
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                
            elif((rgb_cols[k][1] - 127) * 3 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (4.8) + 127 > rgb_cols[k][2]): #Gold
                rgb_cols[k][0] = 21
                rgb_cols[k][1] = 130
                rgb_cols[k][2] = 200
                
            elif((rgb_cols[k][1] - 127) * 4.8 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3) + 127 < rgb_cols[k][2]): #Yellow
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 220
                
            elif((rgb_cols[k][1] - 127) * (-3) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 127 < rgb_cols[k][2]): #Green
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                
            elif((rgb_cols[k][1] - 127) * 0.35 + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.6) + 127 > rgb_cols[k][2]): #Mizu (water)
                rgb_cols[k][0] = 20
                rgb_cols[k][1] = 112
                rgb_cols[k][2] = 103
                
            elif((rgb_cols[k][1] - 127) * (-1.6) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.3) + 127 > rgb_cols[k][2]): #Purple
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                
            elif((rgb_cols[k][1] - 127) * (-0.3) + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.6 + 127 > rgb_cols[k][2]): #Pink
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                
            elif((rgb_cols[k][1] - 127) * 0.6 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3 + 127 > rgb_cols[k][2]): #Orange
                rgb_cols[k][0] = 18
                rgb_cols[k][1] = 180
                rgb_cols[k][2] = 200
            
        elif(21 <= rgb_cols[k][0] <= 24): # High luminance
            if(((rgb_cols[k][1] - 127)**2) + ((rgb_cols[k][2] - 127)**2) >  50 and (rgb_cols[k][1] - 127)* 0.2 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 127  >= rgb_cols[k][2]): #Green
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130

            elif(((rgb_cols[k][1] - 127)**2) + ((rgb_cols[k][2] - 127)**2) <  82): #White
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127

            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 127)**2)) > 1600 and (rgb_cols[k][1] - 127) * 0.6 + 127 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 127 < rgb_cols[k][2]): #Yellow
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 220

            elif((rgb_cols[k][1] - 127)* (3) + 127 > rgb_cols[k][2] and (rgb_cols[k][1] - 127)*(0.6) + 127 < rgb_cols[k][2]): #Hada (Skin)
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
               
            elif((rgb_cols[k][1] - 127)*(0.6) + 127 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)*(-0.3) + 127 <= rgb_cols[k][2]): #Pink
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                
            elif((rgb_cols[k][1] - 127)*(-1.7) + 127 < rgb_cols[k][2] and (rgb_cols[k][1] - 127)*(-0.3) + 127 > rgb_cols[k][2]): #Purple
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                
            elif((rgb_cols[k][1] - 127)* (3) + 127 < rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 127 < rgb_cols[k][2]): #Cream
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 150
                
            elif((rgb_cols[k][1] - 127)*(-1.7) + 127 > rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.2 + 127 > rgb_cols[k][2] ): #Mizu (water)
                rgb_cols[k][0] = 20
                rgb_cols[k][1] = 112
                rgb_cols[k][2] = 103
                
        elif(rgb_cols[k][0] > 24): #High luminance
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                

    # Label to RGB image
    clustered_colors = rgb_cols[cluster_labels]
    clustered_image = np.reshape(clustered_colors, (h, w, c))
    # print(clustered_image.shape)

    af_img_L,af_img_a,af_img_b = cv2.split(clustered_image)
    clustered_image[:,:,0] = af_img_L * 10

    img_Lab2 = cv2.cvtColor(clustered_image.astype(np.uint8), cv2.COLOR_Lab2BGR)
    cl_renketu = cv2.hconcat([img_r, img_Lab2])
    cv2.imwrite(title+'label_' + ext, cl_renketu)
    print(title+ext)
