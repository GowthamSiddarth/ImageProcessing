import cv2
from showimage import show_image

img = cv2.imread('img/bookpage.jpg')
show_image('bookpage', img)

threshold = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
show_image('threshold', threshold)
