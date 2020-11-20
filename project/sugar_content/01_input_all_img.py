
#핵심: 사각형안에 숫자 넣어 저장.
# 1. dataset폴더에서 이미지데이터(흰색배경, 검정글씨)를 바꾸기
# 2. 글씨에 해당하는 부분만 사각프레임으로 찾음.
# 3. 사각형프레임 부분만 저장.

import cv2
import os


groups_folder_path = './data/dataset5/'
categories = ['0','1','2','3','4','5','6','7','8','9']
num_classes = len(categories)
count = 0

image_w = 28
image_h = 28
img_list = []

for idex, categorie in enumerate(categories):
    image_dir = groups_folder_path + categorie + '/'
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(image_dir + filename)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

            ret, img_th = cv2.threshold(img_blur, 160, 255, cv2.THRESH_BINARY_INV)
            image, contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            rects = [cv2.boundingRect(each) for each in contours]

            img_result = []
            img_for_class = img.copy()
            margin_pixel = 0

            for rect in rects:
                if 0 < rect[0] < 50:
                    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 2)
                    # plt.imshow(img)
                    # plt.show()

                    img_result.append(
                        img_for_class[rect[1] - margin_pixel: rect[1] + rect[3] + margin_pixel,
                        rect[0] - margin_pixel: rect[0] + rect[2] + margin_pixel])

                    count += 1
                img_list.append(img_result)


for x in range(len(img_list)):
    name = 'result_' + str(x) + '.jpg'
    if x != 355 and x != 464:
        save = cv2.imwrite(name, img_list[x][0])

print('이미지데이터 개수:',count)




