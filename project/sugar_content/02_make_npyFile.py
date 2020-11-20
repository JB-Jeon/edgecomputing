
# 데이터셋에 저장된 이미지를 읽어와서 28x28 크기로 변경 후 딥러닝할 수 있게 npy파일로 저장함.

import cv2
import os, re, glob
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

groups_folder_path = './data/dataset3_RF/'
categories = ['0','1','2','3','4','5','6','7','8','9']
num_classes = len(categories)
count = 0

image_w = 28
image_h = 28
X = []
Y = []

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + categorie + '/'

    for top, dir, f in os.walk(image_dir):
        for filename in f:
            print(image_dir + filename)
            img = cv2.imread(image_dir + filename, cv2.IMREAD_GRAYSCALE)
            ret2, dst = cv2.threshold(img, 200, 255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.resize(dst, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])

            # plt.figure(figsize=(5, 5))
            # plt.subplot(1, 2, 1) #원본
            # plt.imshow(dst)
            # plt.subplot(1, 2, 2) #사이즈 변경한 이미지
            # plt.imshow(img)
            # plt.show()
            count += 1

            X.append(img / 256)
            Y.append(label)


X = np.array(X)
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X,Y)
xy = (X_train, X_test, Y_train, Y_test)

np.save("./img_data.npy", xy)
print('이미지데이터 개수:',count)

