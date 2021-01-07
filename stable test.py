import cv2, dlib
import numpy as np
import math, sys
PREDICTOR_PATH="shape_predictor_70_face_landmarks.dat"
RESIZE_HEIGHT = 360
NUM_FRAMES_FOR_FPS = 100
SKIP_FRAMES = 3


def interEyeDistance(predict):
  leftEyeLeftCorner = (predict[36].x, predict[36].y)
  rightEyeRightCorner = (predict[45].x, predict[45].y)
  distance = cv2.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner))
  distance = int(distance)
  return distance


winName = "Stabilized facial landmark detector"
cv2.namedWindow(winName, cv2.WINDOW_NORMAL)

# Initializing video capture object.
path = "C:/Users/Mayank Vikram/Documents/Computer Vision/week9/data/project/taarefinal.mp4"
cap = cv2.VideoCapture(0)
# Error message if camera fails to open.
if (cap.isOpened() == False):
    print("Unable to connect to camera")

winSize = 50
maxLevel = 5
fps = 30.0
# Grab a frame
ret, imPrev = cap.read()

# Convert to grayscale.
imGrayPrev = cv2.cvtColor(imPrev, cv2.COLOR_BGR2GRAY)

# Finding the size of the image.
size = imPrev.shape[0:1]

detector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# Initializing the parameters
points = []
pointsPrev = []
pointsDetectedCur = []
pointsDetectedPrev = []

eyeDistanceNotCalculated = True
eyeDistance = 0
isFirstFrame = True
# Initial value, actual value calculated after 100 frames
fps = 10
showStabilized = False
count = 0
while (True):
    if (count == 0):
        t = cv2.getTickCount()

    # Grab a frame
    ret, im = cap.read()
    # COnverting to grayscale
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    height = im.shape[0]
    IMAGE_RESIZE = float(height) / RESIZE_HEIGHT
    # Resize image for faster face detection
    imSmall = cv2.resize(im, None, fx=1.0 / IMAGE_RESIZE, fy=1.0 / IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

    # Skipping the frames for faster processing
    if (count % SKIP_FRAMES == 0):
        faces = detector(imSmall, 0)

    # If no face was detected
    if len(faces) == 0:
        print("No face detected")

    # If faces are detected, iterate through each image and detect landmark points
    else:
        for i in range(0, len(faces)):
            # print("face detected")
            # Face detector was found over a smaller image.
            # So, we scale face rectangle to correct size.
            newRect = dlib.rectangle(int(faces[i].left() * IMAGE_RESIZE),
                                     int(faces[i].top() * IMAGE_RESIZE),
                                     int(faces[i].right() * IMAGE_RESIZE),
                                     int(faces[i].bottom() * IMAGE_RESIZE))
            # Handling the first frame of video differently,for the first frame copy the current frame points
            if (isFirstFrame == True):
                pointsPrev = []
                pointsDetectedPrev = []
                [pointsPrev.append((p.x, p.y)) for p in landmarkDetector(im, newRect).parts()]
                [pointsDetectedPrev.append((p.x, p.y)) for p in landmarkDetector(im, newRect).parts()]

            # If not the first frame, copy points from previous frame.
            else:
                pointsPrev = []
                pointsDetectedPrev = []
                pointsPrev = points
                pointsDetectedPrev = pointsDetectedCur

            # pointsDetectedCur stores results returned by the facial landmark detector
            # points stores the stabilized landmark points
            points = []
            pointsDetectedCur = []
            [points.append((p.x, p.y)) for p in landmarkDetector(im, newRect).parts()]
            [pointsDetectedCur.append((p.x, p.y)) for p in landmarkDetector(im, newRect).parts()]

            # Convert to numpy float array
            pointsArr = np.array(points, np.float32)
            pointsPrevArr = np.array(pointsPrev, np.float32)

            # If eye distance is not calculated before
            if eyeDistanceNotCalculated:
                eyeDistance = interEyeDistance(landmarkDetector(im, newRect).parts())
                # print(eyeDistance)
                eyeDistanceNotCalculated = False

            if eyeDistance > 100:
                dotRadius = 3
            else:
                dotRadius = 2

            print(eyeDistance)
            sigma = eyeDistance * eyeDistance / 400
            s = 2 * int(eyeDistance / 4) + 1

            #  Set up optical flow params
            lk_params = dict(winSize=(s, s), maxLevel=5,
                             criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

            pointsArr, status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev, imGray, pointsPrevArr, pointsArr, **lk_params)
            sigma = 100

            # Converting to float
            pointsArrFloat = np.array(pointsArr, np.float32)

            # Converting back to list
            points = pointsArrFloat.tolist()

            # Final landmark points are a weighted average of
            # detected landmarks and tracked landmarks
            for k in range(0, len(landmarkDetector(im, newRect).parts())):
                d = cv2.norm(np.array(pointsDetectedPrev[k]) - np.array(pointsDetectedCur[k]))
                alpha = math.exp(-d * d / sigma)
                points[k] = (1 - alpha) * np.array(pointsDetectedCur[k]) + alpha * np.array(points[k])

            # Drawing over the stabilized landmark points
            if showStabilized is True:
                for p in points:
                    cv2.circle(im, (int(p[0]), int(p[1])), dotRadius, (255, 0, 0), -1)
            else:
                for p in pointsDetectedCur:
                    cv2.circle(im, (int(p[0]), int(p[1])), dotRadius, (0, 0, 255), -1)

            isFirstFrame = False
            count = count + 1

            # Calculating the fps value
            if (count == NUM_FRAMES_FOR_FPS):
                t = (cv2.getTickCount() - t) / cv2.getTickFrequency()
                fps = NUM_FRAMES_FOR_FPS / t
                count = 0
                isFirstFrame = True

            # Display the landmarks points
            cv2.putText(im, "{}-fps".format(fps), (75, size[0] - 75), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180, 0, 0), 2)
            cv2.imshow(winName, im)
            key = cv2.waitKey(25) & 0xFF

            # Use spacebar to toggle between Stabilized and Unstabilized version.
            if key == 32:
                showStabilized = not showStabilized

            # Stop the program.
            if key == 27:  # ESC
                cv2.destroyAllwindows()
                cap.release()
            # sys.exit()
            # Getting ready for next frame
            imPrev = im
            imGrayPrev = imGray