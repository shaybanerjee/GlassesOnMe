import argparse
import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import time
import dlib 
import numpy as np

# notes: cv2 is the opencv packages
# VideoStream creates a videostream thats on thread
# face_utils translates output from dlib to something we can 
#    use with opencv and numpy
# resize allows us to resize webcam to high resolutions

parser = argparse.ArgumentParser()
parser.add_argument("-predictor", required=True, help="patch to predictor")
args = parser.parse_args()

videoStream = VideoStream()
vs = videoStream.start()

# waiting for the webcam to open
time.sleep(1.5)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)

while True:
    frame = vs.read()
    frame = resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    for rect in rects:
        x, y, w, h = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (225, 128, 0), 2)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        for i in shape[36:48]:
            cv2.circle(frame, tuple(i), 2, (128, 255, 0))
    cv2.imshow("face detect", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if (key == ord("q")):
        break

cv2.destroyAllWindows()
vs.stop()
