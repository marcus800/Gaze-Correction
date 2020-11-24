# We Import the necessary packages needed
import cv2
import numpy as np
import dlib

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("test_video.mp4")




import face_recognition

test = True


while True:
    ret, frame = cap.read()
    # Resize frame of video to 1/4 size for faster face recognition processing
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small[:, :, ::-1]
    rgb_small_frame = small[:, :, ::-1]


    # face_locations = face_recognition.face_locations(rgb_small_frame)
    # face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame)
    face_landmarks_list = face_recognition.face_landmarks(rgb_small_frame,model="large")


    if not face_landmarks_list:
        cv2.putText(frame,"not found",(20,40),cv2.FONT_HERSHEY_SIMPLEX,color=(255, 0, 0),fontScale=1)

    # print(face_landmarks_list)
    # print(face_locations)

    if face_landmarks_list:
        for (x,y) in face_landmarks_list[0]['left_eye']:
            cv2.circle(frame, (x*2, y*2), 2, (255, 255, 0), -1)

        for (x,y) in face_landmarks_list[0]['right_eye']:
            cv2.circle(frame, (x*2, y*2), 2, (255, 255, 0), -1)



    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break  # press esc the frame is destroyed