
#28x28 크기의 숫자값 1개 읽어서 딥러닝을 통해 예측한 lcd숫자값 출력

import cv2
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

img = cv2.imread('./result_0.jpg', cv2.IMREAD_GRAYSCALE)
ret2, img_gray = cv2.threshold(img, 200, 255,
                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
plt.figure(figsize=(5,5))
plt.imshow(img_blur)
plt.show()

plt.imshow(cv2.resize(img_blur,(28,28)))
plt.show()

from keras.models import load_model
model = load_model('./model_lcd/37-0.3457.hdf5')

test_num = cv2.resize(img_blur, (28,28))
test_num = (test_num > 100) * test_num
test_num = test_num.astype('float32') / 255.

plt.imshow(test_num, cmap='Greys', interpolation='nearest')
test_num = test_num.reshape((1, 28, 28, 1))
print('The Answer is ', model.predict_classes(test_num))