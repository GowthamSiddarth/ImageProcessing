import cv2
from matplotlib import pyplot as plt

matching_img = cv2.imread('img/opencv-feature-matching-image.jpg', 0)
matching_template = cv2.imread('img/opencv-feature-matching-template.jpg', 0)

orb = cv2.ORB_create()

key_points_img, desc_img = orb.detectAndCompute(matching_img, None)
key_points_template, desc_template = orb.detectAndCompute(matching_template, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(desc_img, desc_template)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(matching_img, key_points_img, matching_template, key_points_template, matches[:10], None,
                       flags=2)
plt.imshow(img3)
plt.show()
