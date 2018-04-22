import cv2
from showimage import show_image

img = cv2.imread('img/bookpage.jpg')
show_image('bookpage', img)

threshold_without_grayscale = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)[1]
show_image('threshold_without_grayscale', threshold_without_grayscale)

img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
show_image('img2gray', img2gray)

threshold_with_grayscale = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)[1]
show_image('threshold_with_grayscale', threshold_with_grayscale)

adaptive_threshold = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 1)
show_image('adaptive_threshold', adaptive_threshold)

otsu_threshold = cv2.threshold(img2gray, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
show_image('otsu_threshold', otsu_threshold)
