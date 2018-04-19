import cv2
import numpy as np

color_img = cv2.imread('img/watch.jpg', cv2.IMREAD_COLOR)
cv2.imshow("color_image", color_img)

cv2.line(img=color_img, pt1=(0, 0), pt2=(150, 150), color=(255, 255, 255), thickness=15)
cv2.imshow("line_color_image", color_img)

cv2.rectangle(img=color_img, pt1=(15, 25), pt2=(200, 148), color=(0, 0, 255), thickness=15)
cv2.imshow("rect_color_image", color_img)

cv2.circle(img=color_img, center=(100, 65), radius=55, thickness=-1, color=(255, 255, 0))
cv2.imshow("circle_color_image", color_img)

pts = np.array([[10, 2], [23, 27], [23, 53], [32, 12], [31, 12]], dtype=np.int32)
cv2.polylines(img=color_img, pts=[pts], isClosed=True, color=(255, 0, 255), thickness=15)
cv2.imshow("polygon_color_image", color_img)

font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img=color_img, text="OpenCV", org=(0, 130), fontFace=font, fontScale=1, color=(255, 255, 255))
cv2.imshow("text_color_image", color_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
