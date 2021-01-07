# We Import the necessary packages needed
import cv2
import numpy as np
import dlib
import ctypes
import os
import time
import math
from sklearn.linear_model import LinearRegression
import pyvirtualcam


# cap = cv2.VideoCapture("data/LightingVideo480.avi")
cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)

# Doesnt seem to do anything
cap.set(cv2.CAP_PROP_FPS, 30)
WIDTH = int(cap.get(3))
HIGHT = int(cap.get(4))


def run():
    intraFace = loaddll()

    tic = time.time()
    # init setup C
    _, frame = cap.read()

    X = np.zeros(dtype=ctypes.c_float, shape=(2, 49))
    X0 = np.zeros(dtype=ctypes.c_float, shape=(2, 49))
    intraFace.init(frame.shape[0], frame.shape[1], frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                   X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), X0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    prePoints = X0.copy()
    prePoints = prePoints.reshape((49 * 2), order='F')
    prePoints = prePoints.reshape((49, 2), order='A')
    prePoints = prePoints[20:32]

    done = True
    preFrame = frame.copy()
    while done:
        # About -0.15 if cant detect
        # 0.01'0.008 if not mostly fast
        val = intraFace.detect(frame.shape[0], frame.shape[1], frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))

        # eyeCords, noseCords = drawpoints(frame, X0, val)

        if val == 1:

            prePoints, eyeCords = stablepoints(frame,preFrame,prePoints,X0)
            boudingbox(frame, eyeCords, 1)
            box = boudingbox(frame, eyeCords, 2)
            bigbox = boudingbox(frame, eyeCords, 4)
            # noseline(frame, noseCords)
            # reprinteyes(frame, box, bigbox)
            frame2 = colourCorrectAndFrameBlend(frame,box,bigbox,eyeCords)
        else:
            frame2 = frame
        # big = cv2.resize(frame, None, fx=2, fy=2)
        cv2.imshow("Frame", frame)
        cv2.imshow("Frame2", frame2)

        # About 0.032 to here
        #0.002 ish
        k = cv2.waitKey(1)
        if k == 27:
            break
        # if val ==1:
        if k == 32:
            # Saving Images
            for i in range(2):
                eyeCord = box[i]
                sbox = np.int0(eyeCord)
                biCord = bigbox[i]
                Bbox = np.int0(biCord)
                (Btopx, Btopy) = (np.min(Bbox[:, 0]), np.min(Bbox[:, 1]))
                (Bbotx, Bboty) = (np.max(Bbox[:, 0]), np.max(Bbox[:, 1]))
                Beye = (frame[Btopy:Bboty, Btopx:Bbotx]).copy()
                boundInB = sbox - [Btopx, Btopy]
                np.save('savedImages\BeyeCords'+str(i)+'.npy',boundInB)
                cv2.imwrite(os.path.abspath("savedImages")+"\Beye"+str(i)+".bmp", img=Beye)
                cv2.imwrite(os.path.abspath("savedImages")+"\face"+".bmp", img=frame)

        preFrame = frame
        done, frame = cap.read()

        # with pyvirtualcam.Camera(width=WIDTH, height=HIGHT, fps=30) as cam:
        #     rgba = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
        #     cam.send(rgba)



    toc = time.time()
    print(f"Total time: {toc - tic:0.4f} seconds")



def noseline(frame, noseCords):
    cv2.line(frame, noseCords[0], noseCords[1], (0, 0, 0), 3)
    cv2.line(frame, noseCords[1], noseCords[2], (0, 0, 0), 3)
    cv2.line(frame, noseCords[2], noseCords[3], (0, 0, 0), 3)
    return


def colourCorrectAndFrameBlend(frame, boxcords, bigboxcords,eyeCords):
    frame = frame.copy()
    for i in range(2):
        #Only doing bigEye
        boxcord = boxcords[i]
        box = np.int0(boxcord)
        biCord = bigboxcords[i]
        Bbox = np.int0(biCord)
        eyebox = np.int0(eyeCords[i])

        (Btopx, Btopy) = (np.min(Bbox[:, 0]), np.min(Bbox[:, 1]))
        (Bbotx, Bboty) = (np.max(Bbox[:, 0]), np.max(Bbox[:, 1]))

        Beye = (frame[Btopy:Bboty, Btopx:Bbotx]).copy()

        boundInB = box - [Btopx, Btopy]
        BsavedEye = cv2.imread(os.path.abspath("savedImages") + "\Beye" + str(i) + ".bmp")
        BsavedCords = np.load('savedImages\BeyeCords' + str(i) + '.npy')

        #Creating a matrix to warp the saved eyes corner points to the current eye points
        warp_mat = cv2.getAffineTransform(BsavedCords[:3].astype(np.float32), boundInB[:3].astype(np.float32))
        Bfittedeye = cv2.warpAffine(BsavedEye, warp_mat, (Beye.shape[1], Beye.shape[0]))

        #TIME START HERE

        #Making predicted eyes:
        bpredeyes1 = Beye.copy()
        bpredeyes2 = Beye.copy()
        for c in range(3):
            bpredeyes1[:, :, c] = fitEye(Beye[:, :, c])
            # bpredeyes1[:, :, c] = fitEye(Bfittedeye[:, :, c])
            bpredeyes2[:, :, c] = fitEye(Bfittedeye[:, :, c])

        bdiv = np.divide(bpredeyes1, bpredeyes2, where=bpredeyes2 != 0)
        #Maybe shouldn't?
        bound(bdiv)
        doublesizeshow("div",bdiv.astype(np.uint8))
        doublesizeshow("p1",bpredeyes1)
        doublesizeshow("p2",bpredeyes2)
        bcolourcorrected = Bfittedeye * bdiv
        #Bounding
        bound(bcolourcorrected)
        bcolourcorrected = np.uint8(bcolourcorrected)
        #to here about 0.2

        # Say what type of blending to be used:
        # ff- full frame laplcisain,
        # el- just the eye laplcisain,
        # b- normal blur edges
        # n- no blending just overlaying with mask
        blendingType = "blurRound"



        #This blending is kinda slow but intrestly works quite well even without coulor correcting
        # Time about - -.02/3
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fmask = np.zeros_like(gray)
        cv2.drawContours(fmask, [box], 0, 255, -1)  # Draw filled contour in mask
        Fmask2 = np.zeros_like(gray)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(Fmask2, [box], 0, 255, 2)  # Draw filled contour in mask
        Fmask3 = cv2.bitwise_and(fmask, Fmask2)
        fmask = fmask - Fmask3
        innermask = fmask[Btopy:Bboty, Btopx:Bbotx]


        if blendingType == "ff":
            ceye = np.zeros_like(frame)
            # Only replacing the eyes
            ceye[Btopy:Bboty, Btopx:Bbotx] = bcolourcorrected

            frame = laplacianSameSize(frame, ceye, fmask, 4)


        #This works well also
        #Blending just eye section:
        # Time about -0.001/0.00009
        if blendingType == "el":
            # doublesizeshow("fittedeye",Bfittedeye)
            beyeblend = laplacianSameSize(Beye,bcolourcorrected,innermask,3)
            frame[Btopy:Bboty, Btopx:Bbotx] = beyeblend


        #linear bluring:
        # Time about -.002
        if blendingType == "blur":
            blur = cv2.GaussianBlur(innermask, (11, 11), 20)
            for c in range(3):
                frame[Btopy:Bboty, Btopx:Bbotx,c] = (frame[Btopy:Bboty, Btopx:Bbotx,c]*(1-blur/255) + bcolourcorrected[:,:,c]*(blur/255)).astype(np.uint8)

        if blendingType == "blurRound":
            rmask1 = np.zeros_like(innermask)
            rmask2 = np.zeros_like(innermask)
            cv2.drawContours(rmask1, [eyebox - [Btopx, Btopy]], 0, 255, -1)
            cv2.drawContours(rmask2, [eyebox], 0, 255, 2)
            rmask3 = cv2.bitwise_and(rmask1, rmask2)
            rmask = rmask1 - rmask3
            blur = cv2.GaussianBlur(rmask, (15, 15), 30)
            doublesizeshow("newbox",rmask)
            doublesizeshow("blur",blur)
            rmask1[rmask == 0] = blur[rmask == 0]
            doublesizeshow("outblur",rmask1)
            # blur = cv2.GaussianBlur()
            for c in range(3):
                frame[Btopy:Bboty, Btopx:Bbotx,c] = (frame[Btopy:Bboty, Btopx:Bbotx,c]*(1-rmask1/255) + bcolourcorrected[:,:,c]*(rmask1/255)).astype(np.uint8)




        if blendingType == "n":
            frame[Btopy:Bboty, Btopx:Bbotx][innermask== 255] = bcolourcorrected[innermask== 255]


    return frame

def reprinteyes(frame, eyesCords, bigcords):
    for i in range(2):
        eyeCord = eyesCords[i]
        box = np.int0(eyeCord)
        biCord = bigcords[i]
        Bbox = np.int0(biCord)

        (Btopx, Btopy) = (np.min(Bbox[:, 0]), np.min(Bbox[:, 1]))
        (Bbotx, Bboty) = (np.max(Bbox[:, 0]), np.max(Bbox[:, 1]))
        (topx, topy) = (np.min(box[:, 0]), np.min(box[:, 1]))
        (botx, boty) = (np.max(box[:, 0]), np.max(box[:, 1]))



        eye = (frame[topy:boty, topx:botx]).copy()
        Beye = (frame[Btopy:Bboty, Btopx:Bbotx]).copy()
        doublesizeshow("eye", eye)

        boundcords = box - [topx, topy]
        # Bboundcords = Bbox - [Btopx, Btopy]
        boundInB = box - [Btopx, Btopy]

        #Got a few options:
        #1) do whole image with corrected eye mask and ignore black edgues...
        #2) try fix black edges somehow... more face or just blend or whatever
        #3) blend fitted small eye with big other eye!! good idea, but still might have outer ring afterfacts
        #4) blend fitted eye with outer eye and outer eye with overall eye



        # THIS IS BAD - Corrected trying to fit to eye!
        # colourcorrected[mask == 0] = (eye[mask == 0]).copy()
        # cc = laplacianPyramidBlending(frame, eye, colourcorrected, mask, boundcords)
        # doublesizeshow("cc", cc)
        # frame[topy:boty, topx:botx] = cc

        #Whole frame try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Fmask = np.zeros_like(gray)
        cv2.drawContours(Fmask, [box], 0, 255, -1)  # Draw filled contour in mask
        Fmask2 = np.zeros_like(gray)  # Create mask where white is what we want, black otherwise
        cv2.drawContours(Fmask2, [box], 0, 255, 2)  # Draw filled contour in mask
        Fmask3 = cv2.bitwise_and(Fmask, Fmask2)
        Fmask = Fmask - Fmask3
        mask = Fmask[topy:boty, topx:botx]
        #TODO I need to make the cc I pass in bugger so the mask is nicely within it!
        # also have to deal with underflows and overflows

def gaussianPyramid(img, num_levels):
    # lower = np.float32(img.copy())
    lower = img.copy()
    gp = [np.float32(lower)]
    for i in range(num_levels):
        lower = cv2.pyrDown(lower)
        gp.append(np.float32(lower))
    return gp


def laplacianPyramid(gp):
    levels = len(gp) - 1
    lp = [gp[levels]]
    for i in range(levels, 0, -1):
        size = (gp[i - 1].shape[1], gp[i - 1].shape[0])
        GE = cv2.pyrUp(gp[i], dstsize=size)
        L = gp[i - 1] - GE
        lp.append(L)
    return lp

def laplacianSameSize(outerImage, innerImage, mask, levels):

    gpCEye = gaussianPyramid(innerImage, levels)
    lpCEye = laplacianPyramid(gpCEye)
    gpFrame = gaussianPyramid(outerImage, levels)
    lpFrame = laplacianPyramid(gpFrame)

    gpMask = gaussianPyramid(mask, levels)

    gpMask.reverse()
    LS = []
    #Appling the mask
    for lFrame, lCEye, gMask in zip(lpFrame, lpCEye, gpMask):
        lFrame[gMask == 255] = lCEye[gMask == 255]
        LS.append(lFrame)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1,levels+1):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_,dstsize=size)
        ls_ = ls_ + LS[i]

    #Making it above 0 before becoming uint8
    bound(ls_)
    return ls_.astype(np.uint8)


#Until functions
def doublesizeshow(name, im):
    big = cv2.resize(im, None, fx=4, fy=4)
    cv2.imshow(name, big)
def bound(im):
    im[im < 0] = 0
    im[im > 255] = 255

def wait():
    if cv2.waitKey(100000) == 27:
        quit()

def fitEye(eye, mask=None):
    X = np.arange(eye.shape[0])
    Y = np.arange(eye.shape[1])

    # If sqaure needed
    # N = Y.shape[0] - X.shape[0]
    # X = np.pad(X, (0, N), 'constant')
    # Z = np.pad(eye, ((0, N), (0,0)), 'constant')

    Z = eye.copy()
    OnesX = X ** 0
    OnesY = Y ** 0
    XY = np.outer(X, Y).flatten()

    A = np.array([np.ones_like(XY), np.outer(OnesX, Y).flatten(), np.outer(OnesX, Y ** 2).flatten(),
                  np.outer(OnesX, Y ** 3).flatten(), np.outer(X, OnesY).flatten(), XY,
                  np.outer(X, Y ** 2).flatten(), np.outer(X ** 2, OnesY).flatten(), np.outer(X ** 2, Y).flatten(),
                  np.outer(X ** 3, OnesY).flatten()]).T
    #
    if mask is not None:
        A[mask.flatten() == 0] = 0
        Z[mask == 0] = 0

    Z = Z.flatten()
    clf = LinearRegression()
    clf.fit(A, Z)

    neweye = clf.predict(A)
    #removing negative values and capping
    bound(neweye)
    # Maybe should bound more to remove highlights that don't exist

    neweye = np.uint8(neweye)
    neweye = neweye.reshape((X.shape[0], Y.shape[0]))
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
                cv2.drawContours(frame, [box], 0, (0, 0, 0), 2)
            else:
                cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)
        eyes.append(eye)
    return eyes


def stablepoints(frame,preFrame,pre,points,s= 50):


    points = points.reshape((49 * 2), order='F')
    points = points.reshape((49, 2), order='A')
    points = points[19:31]
    lk_params = dict(winSize=(s, s), maxLevel=5, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    graypre = cv2.cvtColor(preFrame, cv2.COLOR_BGR2GRAY)

    T0 = time.time()
    OFpoints, status, err = cv2.calcOpticalFlowPyrLK(graypre, gray, pre, None, **lk_params)
    # print(time.time()-T0)

    # can make it much faster and better with np..
    for i in range(0,len(points)):
        d = cv2.norm(points[i]-pre[i]+5)
        alpha = math.exp(-d *d / 100)
        # alpha = 0.5
        # print(alpha)
        OFpoints[i] = (1-alpha)* points[i] +alpha*OFpoints[i]

    for point in points:
        x = point[0]
        y = point[1]
        # cv2.circle(frame, (int(float(x)), int(float(y))), 1, (255, 255, 0), -1)
    for point in OFpoints:
        x = point[0]
        y = point[1]
        # cv2.circle(frame, (int(float(x)), int(float(y))), 1, (0, 255, 0), -1)

    return points,[OFpoints[0:6],OFpoints[6:12]]
# should be at an angle
def drawpoints(frame, points, val):
    eyeCords = [[], []]
    noseCords = []

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
                cv2.circle(frame, (int(float(x)), int(float(y))), 2, (255, 255, 0), -1)
                if i < 26:
                    eyeCords[0].append((x, y))
                else:
                    eyeCords[1].append((x, y))
            # cv2.putText(frame,str(i), (int(float(x)),int(float(y))), cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=0.2)
            if 11 <= i < 15:
                noseCords.append((x, y))
    return eyeCords, noseCords


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
