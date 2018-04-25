import cv2
import numpy as np

img = cv2.imread('img/opencv-python-dilation-erosion-tutorial.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(src=hsv_img, lowerb=lower_red, upperb=upper_red)
res = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((5, 5), dtype=np.uint8)

erosion = cv2.erode(mask, kernel, iterations=1)
dilation = cv2.dilate(mask, kernel, iterations=1)

opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('img', img)
cv2.imshow('hsv', hsv_img)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.imshow('erosion', erosion)
cv2.imshow('dilation', dilation)
cv2.imshow('opening', opening)
cv2.imshow('closing', closing)

cv2.waitKey(0)
cv2.destroyAllWindows()
