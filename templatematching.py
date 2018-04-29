import cv2
import numpy as np

img = cv2.imread('img/opencv-template-matching-python-tutorial.jpg')
cv2.imshow('orig', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

template = cv2.imread('img/opencv-template-for-matching.jpg', 0)
h, w = template.shape

res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where(res >= threshold)

for point in zip(*loc[::-1]):
    cv2.rectangle(img, point, (point[0] + w, point[1] + h), (0, 255, 255), 2)

cv2.imshow('gray', gray)
cv2.imshow('template', template)
cv2.imshow('res', res)
cv2.imshow('detected', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
