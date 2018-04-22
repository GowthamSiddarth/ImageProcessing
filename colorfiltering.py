import cv2
import numpy as np
from showimage import show_image

img = cv2.imread('img/opencv-python-filtering-example.png')
show_image('img', img)

hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
show_image('hsv', hsv)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
show_image('mask', mask)

res = cv2.bitwise_and(img, img, mask=mask)
show_image('res', res)
