import cv2

cap = cv2.VideoCapture(0)

while True:
    frame = cap.read()[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('gray_frame', gray)
    cv2.imshow('orig_frame', frame)
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
