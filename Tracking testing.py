# We Import the necessary packages needed
import cv2
import numpy as np
import dlib

cap = cv2.VideoCapture("test_video.mp4")
# We initialise detector of dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap.set(3, 640)
cap.set(4, 480)

print(cap.get(3))

# Start the main program
while True:
    test, frame = cap.read()
    if not test:
        break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # We actually Convert to grayscale conversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # trying small
    smallgray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

    faces = detector(smallgray)
    if faces:
        cv2.putText(frame,"found",(20,40),cv2.FONT_HERSHEY_SIMPLEX,color=(255, 0, 0),fontScale=1)
        face = faces[0]
        # The face landmarks code begins from here
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # Then we can also do cv2.rectangle function (frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        landmarks = predictor(gray, face)
        # We are then accesing the landmark points
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break  # press esc the frame is destroyed