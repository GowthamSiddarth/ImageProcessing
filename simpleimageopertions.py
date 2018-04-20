import cv2

img = cv2.imread('img/watch.jpg')
px = img[34, 45]
print(px)

img[34, 45] = [31, 144, 231]
px = img[34, 45]
print(px)

img[100:150,100:150] = [255, 255, 255]

print(img.shape)
print(img.size)
print(img.dtype)

watch_face = img[37:111, 107:194]
img[0:74, 0:87] = watch_face

cv2.imshow("pixel&roi", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
