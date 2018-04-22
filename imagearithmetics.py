import cv2
from showimage import show_image

img1 = cv2.imread('img/3D-Matplotlib.png')
img2 = cv2.imread('img/mainsvmimage.png')

add = img1 + img2
show_image('Img1 + Img2', add)

add = cv2.add(img1, img2)
show_image('cv2.add(Img1, Img2)', add)

add = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
show_image('cv2.addWeighted(img1, img2)', add)

img3 = cv2.imread('img/mainlogo.png')
rows, cols, channels = img3.shape
roi = img1[:rows, :cols]
show_image('roi', roi)

img3togray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
show_image('img3togray', img3togray)

mask = cv2.threshold(img3togray, 220, 255, cv2.THRESH_BINARY_INV)[1]
show_image('mask', mask)

mask_inv = cv2.bitwise_not(mask)
show_image('mask_inv', mask_inv)

img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
show_image('image1_bg', img1_bg)

img3_fg = cv2.bitwise_and(img3, img3, mask=mask)
show_image('imag3_fg', img3_fg)

dst = cv2.add(img1_bg, img3_fg)
show_image('dst', dst)

img1[:rows, :cols] = dst
show_image('final_image', img1)
