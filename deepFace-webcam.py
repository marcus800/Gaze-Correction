# We Import the necessary packages needed
import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture(0)


from deepface import DeepFace
models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
# DeepFace.verify("img1.jpg", "img2.jpg", model_name = models[0])

# print(DeepFace.functions.)

test = True


while True:
    ret, frame = cap.read()


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