import cv2

gray_img = cv2.imread('img/watch.jpg', cv2.IMREAD_GRAYSCALE)
orig_img = cv2.imread('img/watch.jpg', cv2.IMREAD_UNCHANGED)
cv2.imshow('gray_image', gray_img)
cv2.imshow('orig_image', orig_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
