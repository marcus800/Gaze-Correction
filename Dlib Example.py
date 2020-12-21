# We Import the necessary packages needed
import cv2
import numpy as np
import dlib


import pyvirtualcam



cap = cv2.VideoCapture(0)
# We initialise detector of dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


WIDTH = int(cap.get(3))
HIGHT = int(cap.get(4))

print(WIDTH)
print(HIGHT)

while True:
    _, frame = cap.read()
    print(frame)
    # We actually Convert to grayscale conversion
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # faces = detector(gray)
    # if faces:
    #     cv2.circle(frame, (faces[0].left(), faces[0].top()), 2, (255, 255, 0), -1)
    #     cv2.circle(frame, (faces[0].right(), faces[0].bottom()), 2, (255, 255, 0), -1)
    #     cv2.putText(frame,"found",(20,40),cv2.FONT_HERSHEY_SIMPLEX,color=(255, 0, 0),fontScale=1)
    # else:
    #     cv2.putText(frame,"not found",(20,40),cv2.FONT_HERSHEY_SIMPLEX,color=(255, 0, 0),fontScale=1)

    # for face in faces:
    #     # The face landmarks code begins from here
    #     x1 = face.left()
    #     y1 = face.top()
    #     x2 = face.right()
    #     y2 = face.bottom()
    #     # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    #     landmarks = predictor(gray, face)
    #     # We are then accesing the landmark points
    #     for n in range(0, 68):
    #         x = landmarks.part(n).x
    #         y = landmarks.part(n).y
    #         cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    # print(frame)
    rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

    # cam.sleep_until_next_frame()



    if key == 27:
        break  # press