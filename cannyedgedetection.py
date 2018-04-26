import cv2
import numpy as np

img = cv2.imread('img/opencv-canny-edge-detection.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv_img, lowerb=lower_red, upperb=upper_red)
res = cv2.bitwise_and(img, img, mask=mask)
canny_edge = cv2.Canny(img, 100, 200)

cv2.imshow('orig', img)
cv2.imshow('hsv', hsv_img)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.imshow('canny_edge', canny_edge)

cv2.waitKey(0)
cv2.destroyAllWindows()
