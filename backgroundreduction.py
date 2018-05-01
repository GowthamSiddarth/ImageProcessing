import cv2

capture = cv2.VideoCapture(0)
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

while True:
    frame = capture.read()[1]

    fg_mask = bg_subtractor.apply(frame)

    cv2.imshow('orig', frame)
    cv2.imshow('masked', fg_mask)

    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
