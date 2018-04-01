import argparse
import cv2
from imutils.video import VideoStream
from imutils import face_utils, translate, resize
import time
import dlib 
import numpy as np
from scipy.spatial import distance as dist
from scipy.spatial import ConvexHull

# notes: cv2 is the opencv packages
# VideoStream creates a videostream thats on thread
# face_utils translates output from dlib to something we can 
#    use with opencv and numpy
# resize allows us to resize webcam to high resolutions

FULL_POINTS = list(range(0, 68))
FACE_POINTS = list(range(17, 68))
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))  


parser = argparse.ArgumentParser()
parser.add_argument("-predictor", required=True, help="patch to predictor")
args = parser.parse_args()

videoStream = VideoStream()
vs = videoStream.start()

# waiting for the webcam to open
time.sleep(1.5)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor)

def eye_size(eye):
    eyeWidth = dist.euclidean(eye[0], eye[3])
    hull = ConvexHull(eye)
    eyeCenter = np.mean(eye[hull.vertices, :], axis=0)
    eyeCenter = eyeCenter.astype(int)

    return int(eyeWidth), eyeCenter

def place_frame(frame, eyeCenter_left, eyeSize_left, eyeCenter_right, eyeSize_right):
    eyeSize_left = int(eyeSize_left * 1.5)
    eyeSize_right = int(eyeSize_right * 1.5)
    x1_left = int(eyeCenter_left[0,0] - (eyeSize_left/2))  
    x2_left = int(eyeCenter_left[0,0] + (eyeSize_left/2))  
    y1_left = int(eyeCenter_left[0,1] - (eyeSize_left/2))  
    y2_left = int(eyeCenter_left[0,1] + (eyeSize_left/2))  

    x1_right = int(eyeCenter_right[0,0] - (eyeSize_right/2))  
    x2_right = int(eyeCenter_right[0,0] + (eyeSize_right/2))  
    y1_right = int(eyeCenter_right[0,1] - (eyeSize_right/2))  
    y2_right = int(eyeCenter_right[0,1] + (eyeSize_right/2)) 

    h, w = frame.shape[:2]

    if x1_left < 0:  
        x1_left = 0  
    if y1_left < 0:  
        y1_left = 0  
    if x2_left > w:  
        x2_left = w  
    if y2_left > h:  
        y2_left = h 

    if x1_right < 0:  
        x1_right = 0  
    if y1_right < 0:  
        y1_right = 0  
    if x2_right > w:  
        x2_right = w  
    if y2_right > h:  
        y2_right = h 

    # re-calculate the size to avoid clipping  
    eyeOverlayWidth_left = x2_left - x1_left  
    eyeOverlayHeight_left = y2_left - y1_left 

    eyeOverlayWidth_right = x2_right - x1_right  
    eyeOverlayHeight_right = y2_right - y1_right


#----------------------------------------------------------------------------------------
# Load and pre-process frame example
# ---------------------------------------------------------------------------------------

# Load frame image
glass_frame = cv2.imread('./img/glasses.png', -1)
# Create mask using the frame
orig_mask = glass_frame[:,:,3]
# Create the inverted mask for the overlay image
orig_mask_inv = cv2.bitwise_not(orig_mask)
# Convert overlay image to BGR
glass_frame = glass_frame[:, :, 0:3]
origFrameHeight, origFrameWidth = glass_frame.shape[:2]


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
        for i in shape:
            cv2.circle(frame, tuple(i), 2, (128, 255, 0))
        
    cv2.imshow("face detect", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if (key == ord("q")):
        break

cv2.destroyAllWindows()
vs.stop()