# -*- coding: UTF-8 -*-
import os
import glob
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter

# files = glob.glob('./wine/*.jpg')
# files=glob.glob('./store/example_for_paper/*.png')
files = glob.glob('/home/saito/SandBox/Pytorch/clus/kmeans_label/kmeans_label/store/dress/re_dress.png')
f = 0
for f in files:
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
    # print('平坦化後', data_points.shape)
    # K-Means法でクラスタリング(代表的な色を決める)
    kmeans_model = KMeans(n_clusters=25) # クラスタ数を指定
    cluster_labels = kmeans_model.fit_predict(data_points)

    # 整数に変換
    rgb_cols = kmeans_model.cluster_centers_.round(0).astype(int)

    for k in range(25):
        if(rgb_cols[k][0]<2): #黒
            rgb_cols[k][0] = 0
            rgb_cols[k][1] = 127
            rgb_cols[k][2] = 127
            print("黒")
        elif( 2<= rgb_cols[k][0] < 8): #低輝度
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  200): #黒
                rgb_cols[k][0] = 0
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                print("黒")
            elif((rgb_cols[k][1] - 127) * (-0.2) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.7 + 114 > rgb_cols[k][2]): #えんじ
                rgb_cols[k][0] = 7
                rgb_cols[k][1] = 170
                rgb_cols[k][2] = 150
                print("えんじ")
            elif((rgb_cols[k][1] - 127) * 0.7 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-4.8) + 114 < rgb_cols[k][2]): #茶
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 145
                rgb_cols[k][2] = 157
                print("茶")
            elif((rgb_cols[k][1] - 127) * (-4.8) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3.8) + 114 < rgb_cols[k][2]): #抹茶
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 110
                rgb_cols[k][2] = 160
                print("抹茶")
            elif((rgb_cols[k][1] - 127) * (-3.8) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.3 + 114 < rgb_cols[k][2]): #緑
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                print("緑")
            elif((rgb_cols[k][1] - 127) * 1.3 + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1) + 114 > rgb_cols[k][2]): #紺
                rgb_cols[k][0] = 7
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 70
                print("紺")
            elif((rgb_cols[k][1] - 127) * (-1) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.2) + 114 > rgb_cols[k][2]): #紫
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                print("紫")
        elif(8<= rgb_cols[k][0] <12): #中輝度
            # if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 150 and (rgb_cols[k][1] - 127) * (-3.5) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 114 < rgb_cols[k][2]): #抹茶
            #     rgb_cols[k][0] = 12
            #     rgb_cols[k][1] = 110
            #     rgb_cols[k][2] = 160
            #     print("抹茶")
            # elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 150 and (rgb_cols[k][1] - 127) * (-2) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.7 + 114 < rgb_cols[k][2]): #緑
            #     rgb_cols[k][0] = 14
            #     rgb_cols[k][1] = 50
            #     rgb_cols[k][2] = 130
            #     print("緑")
            if(((rgb_cols[k][1] - 127)*2) + (((rgb_cols[k][2] - 114)**2)) <  350): #灰色
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                print("灰色")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  1550 and (rgb_cols[k][1] - 127) * 0.2 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.2 + 114 > rgb_cols[k][2]): #肌色
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                print("肌色")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  1550 and (rgb_cols[k][1] - 127) * (-0.2) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.2 + 114 > rgb_cols[k][2]): #ピンク
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                print("ピンク")
            elif((rgb_cols[k][1] - 127) * (-0.2) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.2 + 114 > rgb_cols[k][2]): #赤
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 200
                rgb_cols[k][2] = 180
                print("赤")
            elif((rgb_cols[k][1] - 127) * (1.2) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3.5 + 114 > rgb_cols[k][2]): #茶
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 145
                rgb_cols[k][2] = 157
                print("茶")
            elif((rgb_cols[k][1] - 127) * 3.5 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3.5) + 114 < rgb_cols[k][2]): #黄土
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 170
                print("黄土")
            elif((rgb_cols[k][1] - 127) * (-3.5) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 114 < rgb_cols[k][2]): #抹茶
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 110
                rgb_cols[k][2] = 160
                print("抹茶")
            elif((rgb_cols[k][1] - 127) * (-2) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.7 + 114 < rgb_cols[k][2]): #緑
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                print("緑")
            elif((rgb_cols[k][1] - 127) * 0.7 + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.8) + 114 > rgb_cols[k][2]): #青
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 118
                rgb_cols[k][2] = 50
                print("青")
            elif((rgb_cols[k][1] - 127) * (-1.8) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.2) + 114 > rgb_cols[k][2]): #紫
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                print("紫")
        elif( 12<= rgb_cols[k][0] <15 ): #中輝度
            # if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 150 and (rgb_cols[k][1] - 127) * (-4) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2.5) + 114 < rgb_cols[k][2]): #抹茶
            #     rgb_cols[k][0] = 12
            #     rgb_cols[k][1] = 110
            #     rgb_cols[k][2] = 160
            #     print("抹茶")
            # elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 150 and(rgb_cols[k][1] - 127) * (-2.5) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 114 < rgb_cols[k][2]): #緑
            #     rgb_cols[k][0] = 14
            #     rgb_cols[k][1] = 50
            #     rgb_cols[k][2] = 130
            #     print("緑")
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  350): #灰色
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                print("灰色")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  1550 and (rgb_cols[k][1] - 127) * 3.7 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-4) + 114 < rgb_cols[k][2] ): #黄色
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 150
                print("クリーム")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  1550 and (rgb_cols[k][1] - 127) * 0.3 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.9 + 114 > rgb_cols[k][2]): #肌色
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                print("肌色")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  4800 and (rgb_cols[k][1] - 127) * (-0.3) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (0.3) + 114 > rgb_cols[k][2]): #肌色
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                print("ピンク")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) >= 1400  and (rgb_cols[k][1] - 127) * 4 + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.9) + 114 > rgb_cols[k][2]): #水色
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 118
                rgb_cols[k][2] = 50
                print("青")
            elif((rgb_cols[k][1] - 127) * 3.7 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-4) + 114 < rgb_cols[k][2]): #黄土
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 170
                print("黄土")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 150 and (rgb_cols[k][1] - 127) * (-4) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2.5) + 114 < rgb_cols[k][2]): #抹茶
                rgb_cols[k][0] = 12
                rgb_cols[k][1] = 110
                rgb_cols[k][2] = 160
                print("抹茶")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 150 and(rgb_cols[k][1] - 127) * (-2.5) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 114 < rgb_cols[k][2]): #緑
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                print("緑")
            elif((rgb_cols[k][1] - 127) * 0.35 + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.9) + 114 > rgb_cols[k][2]): #水色
                rgb_cols[k][0] = 20
                rgb_cols[k][1] = 112
                rgb_cols[k][2] = 103
                print("水色")
            elif((rgb_cols[k][1] - 127) * (-1.9) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.3) + 114 > rgb_cols[k][2]): #紫
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                print("紫")
            elif((rgb_cols[k][1] - 127) * (-0.3) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.9 + 114 > rgb_cols[k][2]): #赤
                rgb_cols[k][0] = 13
                rgb_cols[k][1] = 200
                rgb_cols[k][2] = 180
                print("赤")
            elif((rgb_cols[k][1] - 127) * 0.9 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 1.9 + 114 > rgb_cols[k][2]): #オレンジ
                rgb_cols[k][0] = 18
                rgb_cols[k][1] = 180
                rgb_cols[k][2] = 200
                print("オレンジ")
            elif((rgb_cols[k][1] - 127) * 1.9 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3.7 + 114 > rgb_cols[k][2]): #茶
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 145
                rgb_cols[k][2] = 157
                print("茶")
        elif( 15<= rgb_cols[k][0] <21 ): #中輝度
            # if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 50 and (rgb_cols[k][1] - 127) * (-3) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 114 < rgb_cols[k][2]): #緑
            #     rgb_cols[k][0] = 14
            #     rgb_cols[k][1] = 50
            #     rgb_cols[k][2] = 130
            #     print("緑")
            if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  200): #白色
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                print("白色")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  1550 and (rgb_cols[k][1] - 127) * 3 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3) + 114 < rgb_cols[k][2]): #クリーム
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 150
                print("クリーム")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) <  1550 and (rgb_cols[k][1] - 127) * 0.6 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3 + 114 > rgb_cols[k][2]): #肌
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                print("肌色")
            elif((rgb_cols[k][1] - 127) * 3 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (4.8) + 114 > rgb_cols[k][2]): #金
                rgb_cols[k][0] = 21
                rgb_cols[k][1] = 130
                rgb_cols[k][2] = 200
                print("金色")
            elif((rgb_cols[k][1] - 127) * 4.8 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-3) + 114 < rgb_cols[k][2]): #黄
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 220
                print("黄色")
            elif((rgb_cols[k][1] - 127) * (-3) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 114 < rgb_cols[k][2]): #緑
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                print("緑")
            elif((rgb_cols[k][1] - 127) * 0.35 + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-1.7) + 114 > rgb_cols[k][2]): #水色
                rgb_cols[k][0] = 20
                rgb_cols[k][1] = 112
                rgb_cols[k][2] = 103
                print("水色")
            elif((rgb_cols[k][1] - 127) * (-1.7) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-0.3) + 114 > rgb_cols[k][2]): #紫
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                print("紫")
            elif((rgb_cols[k][1] - 127) * (-0.3) + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.6 + 114 > rgb_cols[k][2]): #ピンク
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                print("ピンク")
            elif((rgb_cols[k][1] - 127) * 0.6 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 3 + 114 > rgb_cols[k][2]): #オレンジ
                rgb_cols[k][0] = 18
                rgb_cols[k][1] = 180
                rgb_cols[k][2] = 200
                print("オレンジ")
        elif(21 <= rgb_cols[k][0] <= 24): #高輝度
            # if(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 50 and (rgb_cols[k][1] - 127) * (-3) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.35 + 114 < rgb_cols[k][2]): #緑
            #     rgb_cols[k][0] = 14
            #     rgb_cols[k][1] = 50
            #     rgb_cols[k][2] = 130
            #     print("緑")
            if(((rgb_cols[k][1] - 127)**2) + ((rgb_cols[k][2] - 114)**2) <  200): #白
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                print("白色")
            elif(((rgb_cols[k][1] - 127)**2) + (((rgb_cols[k][2] - 114)**2)) > 1600 and (rgb_cols[k][1] - 127) * 0.6 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 114 < rgb_cols[k][2]): #黄
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 220
                print("黄色")
            elif((rgb_cols[k][1] - 127)* (3) + 114 > rgb_cols[k][2] and (rgb_cols[k][1] - 127)*(0.6) + 114 < rgb_cols[k][2]): #肌色
                rgb_cols[k][0] = 22
                rgb_cols[k][1] = 140
                rgb_cols[k][2] = 160
                print("肌色")
            elif((rgb_cols[k][1] - 127)*(0.6) + 114 >= rgb_cols[k][2] and (rgb_cols[k][1] - 127)*(-0.3) + 114 <= rgb_cols[k][2]): #ピンク
                rgb_cols[k][0] = 24
                rgb_cols[k][1] = 174
                rgb_cols[k][2] = 134
                print("ピンク")
            elif((rgb_cols[k][1] - 127)*(-1.8) + 114 < rgb_cols[k][2] and (rgb_cols[k][1] - 127)*(-0.3) + 114 > rgb_cols[k][2]): #紫
                rgb_cols[k][0] = 10
                rgb_cols[k][1] = 160
                rgb_cols[k][2] = 95
                print("紫")
            elif((rgb_cols[k][1] - 127)* (3) + 114 < rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 114 < rgb_cols[k][2]): #クリーム
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 150
                print("クリーム")
            elif((rgb_cols[k][1] - 127)* 0.2 + 114 <= rgb_cols[k][2] and (rgb_cols[k][1] - 127)* (-2) + 114  >= rgb_cols[k][2]): #緑
                rgb_cols[k][0] = 14
                rgb_cols[k][1] = 50
                rgb_cols[k][2] = 130
                print("緑")
            elif((rgb_cols[k][1] - 127)*(-1.8) + 114 > rgb_cols[k][2] and (rgb_cols[k][1] - 127)* 0.2 + 114 > rgb_cols[k][2] ): #水色
                rgb_cols[k][0] = 20
                rgb_cols[k][1] = 112
                rgb_cols[k][2] = 103
                print("水色")
        elif(rgb_cols[k][0] > 24): #高輝度
                rgb_cols[k][0] = 25
                rgb_cols[k][1] = 127
                rgb_cols[k][2] = 127
                print("白色")

    # クラスターのラベルを画像のピクセルごとにRGBに変換
    clustered_colors = rgb_cols[cluster_labels]
    clustered_image = np.reshape(clustered_colors, (h, w, c))
    # print(clustered_image.shape)

    af_img_L,af_img_a,af_img_b = cv2.split(clustered_image)
    clustered_image[:,:,0] = af_img_L * 10

    img_Lab2 = cv2.cvtColor(clustered_image.astype(np.uint8), cv2.COLOR_Lab2BGR)
    # cl_renketu = cv2.hconcat([img_r, img_Lab2])
    cv2.imwrite(title+'_bb_label' + ext, img_Lab2)
