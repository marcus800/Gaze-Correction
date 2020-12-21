# We Import the necessary packages needed
import cv2
import numpy as np
import dlib
import ctypes
import os
import time
from sklearn.linear_model import LinearRegression

# cap = cv2.VideoCapture("data/LightingVideo480.avi")
cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

cap.set(cv2.CAP_PROP_FPS, 30)
# Doesnt seem to do anything
WIDTH = int(cap.get(3))
HIGHT = int(cap.get(4))


def run():
    intraFace = loaddll()

    tic = time.perf_counter()
    # init setup C
    _, frame = cap.read()

    X = np.zeros(dtype=ctypes.c_float, shape=(2, 49))
    X0 = np.zeros(dtype=ctypes.c_float, shape=(2, 49))
    intraFace.init(frame.shape[0], frame.shape[1], frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                   X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))

    done = True
    while done:
        # res,val = soppySendfram(frame, intraFace)
        val = intraFace.detect(frame.shape[0], frame.shape[1], frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))

        eyeCords = drawpoints(frame, X0, val)

        if val == 1:
            boudingbox(frame, eyeCords, 1)
            box = boudingbox(frame, eyeCords, 2)
            reprinteyes(frame, box)

        big = cv2.resize(frame, None, fx=2, fy=2)
        cv2.imshow("Frame", big)
        if cv2.waitKey(1) == 27:
            break
        done, frame = cap.read()

    toc = time.perf_counter()
    print(f"Total time: {toc - tic:0.4f} seconds")




def reprinteyes(frame, eyesCords):
    for i in range(1):
        eyeCord = eyesCords[i]
        box = np.int0(eyeCord)

        (topx, topy) = (np.min(box[:, 0]), np.min(box[:, 1]))
        (botx, boty) = (np.max(box[:, 0]), np.max(box[:, 1]))

        eye = frame[topy:boty, topx:botx]
        boundcords = box - [topx, topy]

        savedEye = cv2.imread(os.path.abspath("savedImages") + "\eye" + str(i) + ".bmp")
        savedCords = np.load('savedImages\eyeCords' + str(i) + '.npy')

        warp_mat = cv2.getAffineTransform(savedCords[:3].astype(np.float32), boundcords[:3].astype(np.float32))
        fittedeye = cv2.warpAffine(savedEye, warp_mat, (eye.shape[1], eye.shape[0]))

        # TEST SHADING CODE:


        # coeff, r, rank, s = np.linalg.lstsq(A, Z)


        # eye2 = eye.copy()
        # eye2[mask == 255] = fittedeye[mask == 255]
        # eye2[mask2 == 255] = eye[mask2 == 255]
        #

        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, [boundcords], 0, 255, -1)  # Draw filled contour in mask
        mask2 = np.zeros_like(gray)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask2, [boundcords], 0, 255, 2)  # Draw filled contour in mask


        predeyes1 = eye.copy()
        predeyes2 = eye.copy()
        for c in range(3):
            if i ==1:
                predeyes1[:,:,c] = fitEye(eye[:, :, c],mask) # Trying without mask think no diffrence as should be.
            else:
                predeyes1[:,:,c] = fitEye(eye[:, :, c])

            predeyes2[:,:,c] = fitEye(fittedeye[:, :, c],mask)
        div = np.divide(predeyes1, predeyes2, where=predeyes2 != 0)
        colourcorrected = fittedeye * div
        colourcorrected = np.uint8(colourcorrected)


        eye2 = eye.copy()
        eye2[mask == 255] = colourcorrected[mask == 255]
        eye2[mask2 == 255] = eye[mask2 == 255]
        frame[topy:boty, topx:botx] = eye2




        # cv2.imshow("corrected?", colourcorrected)
        # cv2.imshow("eye", eye)
        # if cv2.waitKey(100000) == 27:
        #     quit()

        # Saving Images
    #     np.save('savedImages\eyeCords'+str(i)+'.npy',boundcords)
    #     cv2.imwrite(os.path.abspath("savedImages")+"\eye"+str(i)+".bmp", img=eye)
    # cv2.imwrite(os.path.abspath("savedImages")+"\face"+".bmp", img=frame)


def fitEye(eye, mask=None):
    X = np.arange(eye.shape[0])
    Y = np.arange(eye.shape[1])

    # If sqaure needed
    N = Y.shape[0] - X.shape[0]
    # X = np.pad(X, (0, N), 'constant')
    # Z = np.pad(eye, ((0, N), (0,0)), 'constant')

    Z = eye.copy()
    OnesX = X ** 0
    OnesY = Y ** 0

    A = np.array([np.outer(X, Y).flatten() * 0 + 1, np.outer(OnesX, Y).flatten(), np.outer(OnesX, Y ** 2).flatten(),
                  np.outer(OnesX, Y ** 3).flatten(), np.outer(X, OnesY).flatten(), np.outer(X, Y).flatten(),
                  np.outer(X, Y ** 2).flatten(), np.outer(X ** 2, OnesY).flatten(), np.outer(X ** 2, Y).flatten(),
                  np.outer(X ** 3, OnesY).flatten()]).T
    #
    if mask is not None:
        A[mask.flatten() == 0] = 0
        Z[mask == 0] = 0

    print(Z.shape)
    Z = Z.flatten()
    clf = LinearRegression()
    print(Z.shape)
    print(A.shape)
    clf.fit(A, Z)

    neweye = clf.predict(A)
    neweye = np.uint8(neweye)
    neweye = neweye.reshape((X.shape[0], Y.shape[0]))
    oldeye = Z.reshape((X.shape[0], Y.shape[0]))
    # cv2.imshow("eye", oldeye)
    # cv2.imshow("Feye", neweye)
    return neweye


def boudingbox(frame, eyeCords, scale):
    eyes = []
    for i in range(2):
        eye = np.zeros(shape=(4, 2))
        eyenp = np.array(eyeCords[i])
        v = eyenp[2] - eyenp[4]
        u = eyenp[1] - eyenp[5]
        d = (u + v) / 4
        d = d * scale
        hor = (eyenp[3] - eyenp[0]) / 4
        hord = hor * scale - hor
        eye[0] = eyenp[0] + d - hord
        eye[3] = eyenp[0] - d - hord
        eye[1] = eyenp[3] + d + hord
        eye[2] = eyenp[3] - d + hord

        box = np.int0(eye)
        # Another way to do it but more expensive
        # rect = cv2.minAreaRect(eyenp)
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)

        draw = False
        if draw:
            if scale == 1:
                # cv2.rectangle(frame, (int(eye[0][0]), int(eye[0][1])), (int(eye[3][0]), int(eye[3][1])), (0, 0, 0))
                cv2.drawContours(frame, [box], 0, (0, 0, 0), 2)
            else:
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        eyes.append(eye)
    return eyes


# should be at an angle
def drawpoints(frame, points, val):
    eyeCords = [[], []]

    if val != 1:
        cv2.putText(frame, "Not Found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1)
    else:
        i = 0
        points = points.reshape((49 * 2), order='F')
        points = points.reshape((49, 2), order='A')
        for point in points:
            i += 1
            x = point[0]
            y = point[1]
            if 20 <= i < 32:
                # cv2.circle(frame, (int(float(x)), int(float(y))), 2, (255, 255, 0), -1)
                if i < 26:
                    eyeCords[0].append((x, y))
                else:
                    eyeCords[1].append((x, y))
            # cv2.putText(frame,str(i), (int(float(x)),int(float(y))), cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=0.2)

    return eyeCords


def loaddll():
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_core246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_ffmpeg246_64.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_flann246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_highgui246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_imgproc246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_objdetect246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_features2d246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\opencv_calib3d246.dll'))
    #
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\IntraFaceDLL.dll'))
    # intraFace = ctypes.cdll.LoadLibrary(os.path.abspath('IntraFace_Resources\\IntraFaceTracker.dll'))
    intraFace = ctypes.cdll.LoadLibrary(
        'C:\\Users\\Marcus\\OneDrive\\Documents\\GitHub\\IntraFace\\x64\\Release\\IntraFaceTracker.dll')

    return intraFace


if __name__ == "__main__":
    run()
