# import required library
import cv2
import mediapipe as mp
import numpy as np

# create a class named FaceLandmark
class FaceLandmarks:

    def __init__(self):
        # load Face Mesh to use it to detect facial landmark
        mp_face_mesh = mp.solutions.face_mesh
        # know we will load the face mesh -- object detection --
        self.face_mesh = mp_face_mesh.FaceMesh()

    def get_facial_landmarks(self, frame):
        height, width, _ = frame.shape  # _ because we don't need the channel
        # Know we want to convert the channel of the image from BGR
        # #( that who opencv read the image) to RGP (image in jpg format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(frame_rgb)

        # here we want to find the coordinate of the land mark (the coordinate should be integer not float

        facelandmarks = []
        for facial_landmarks in result.multi_face_landmarks:
            # we need a loop to accesses all the land mark in the face [0, 467] point
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                facelandmarks.append([x, y])

        return np.array(facelandmarks, np.int32)



