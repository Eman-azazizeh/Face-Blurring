# import required library
import cv2
import mediapipe as mp
import numpy as np
from facial_landmarks import FaceLandmarks

# load faceland mark
f1 = FaceLandmarks()

# load the camera or video path
cap = cv2.VideoCapture("walk.mp4")

while True:
    ret, frame = cap.read()
    frame =cv2.resize(frame,(640,640))
    frame_copy = frame.copy()
    height, width, _ = frame.shape

    # Face landmark detection (from the frame we want to get the landmark position)
    landmarks = f1.get_facial_landmarks(frame)

    convexhull = cv2.convexHull(landmarks)
    # surround the landmark by convexhull and polyline
    # Face blurring  full the face with blurring

    # the bound of face here we just one the wight color
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [convexhull], True, 255, 3)
    # now lets fill the convexpoly(fill inside the bound)
    cv2.fillConvexPoly(mask, convexhull, 255)

    # extarct the face from original face
    frame_copy = cv2.blur(frame_copy, (30,30))
    face_extracted = cv2.bitwise_and(frame_copy, frame_copy, mask=mask)

    # the more increse the k-size the more we lose the detail
    blurred_face = cv2.GaussianBlur(face_extracted,(27,27),0)

    # Extract background
    background_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=background_mask)

    # we will add the background the blurring face to gather
    result = cv2.add(background, face_extracted)
    cv2.imshow("reslut", result)

    key = cv2.waitKey(1)



