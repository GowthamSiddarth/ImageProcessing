import cv2


def show_image(window_name, image):
    cv2.imshow(winname=window_name, mat=image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
