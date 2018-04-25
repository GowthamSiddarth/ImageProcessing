import cv2
import numpy as np

img = cv2.imread('img/blurringandsmoothing.png')
hsv_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(src=hsv_img, lowerb=lower_red, upperb=upper_red)
res = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((15, 15), dtype=np.float32) / 225
smoothed = cv2.filter2D(src=res, ddepth=-1, kernel=kernel)

gaussian_blur = cv2.GaussianBlur(res, (15, 15), 0)
median_blur = cv2.medianBlur(res, 15)
bilateral_blur = cv2.bilateralFilter(res, 15, 75, 75)

cv2.imshow(winname='original', mat=img)
cv2.imshow('masked', res)
cv2.imshow('smoothed', smoothed)
cv2.imshow('gaussian_blur', gaussian_blur)
cv2.imshow('median_blur', median_blur)
cv2.imshow('bilateral_blur', bilateral_blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
