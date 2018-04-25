import cv2
import numpy as np

img = cv2.imread('img/opencv-python-gradients.jpg')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv_img, lower_red, upper_red)
res = cv2.bitwise_and(img, img, mask=mask)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

cv2.imshow('orig', img)
cv2.imshow('hsv', hsv_img)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
cv2.imshow('laplacian', laplacian)
cv2.imshow('sobelx', sobelx)
cv2.imshow('sobely', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()
