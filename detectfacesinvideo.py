from imutils.video import VideoStream
from imutils import resize
import cv2
import time
import numpy as np
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-p", "--prototxt", required=True, help="path to Caffe deploy prototxt file")
arg_parser.add_argument("-m", "--model", required=True, help="path to Caffe pre-trained model")
arg_parser.add_argument("-c", "--confidence", type=float, default=0.3,
                        help="minimum probability to filter weak detections")
args = vars(arg_parser.parse_args())

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] starting video stream...")
video_stream = VideoStream(src=0).start()
time.sleep(2)

while not video_stream.stopped:
    frame = video_stream.read()
    frame = resize(frame, width=400)

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")

            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        cv2.destroyAllWindows()
        video_stream.stop()

print("[INFO] Program END")
#python detectfacesinvideo.py -p resources/deploy.prototxt.txt -m resources/res10_300x300_ssd_iter_140000.caffemodel