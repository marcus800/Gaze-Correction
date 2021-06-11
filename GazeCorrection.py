import cv2
import numpy as np
import ctypes
import os
import time
import math
import pyvirtualcam
import scipy.linalg

# To use please downlaod and install OBS, explantion in the readme
VirtualCameraOutput = False

# By default this will be 480p
cap = cv2.VideoCapture(0)

# cap.set(3, 1280)
# cap.set(4, 720)
WIDTH = int(cap.get(3))
HIGHT = int(cap.get(4))
print("Width :", WIDTH)
print("Hight :", HIGHT)


def run():
    intraFace = loaddll()

    _, frame = cap.read()

    X = np.zeros(dtype=ctypes.c_float, shape=(2, 49))
    X0 = np.zeros(dtype=ctypes.c_float, shape=(2, 49))

    angle = np.zeros(dtype=ctypes.c_float, shape=(1, 3))
    rot = np.zeros(dtype=ctypes.c_float, shape=(3, 3))
    intraFace.init(frame.shape[0], frame.shape[1], frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                   X.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   X0.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   angle.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   rot.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                   )

    initiation(X0, angle, intraFace)

    # Loading eyes and camrea pos
    savedEyes = []
    savedCords = []
    for i in range(5):
        savedEyes.append([cv2.imread(os.path.abspath("ReplacementEyes") + "\eye" + "0" + str(i) + ".bmp"),
                          cv2.imread(os.path.abspath("ReplacementEyes") + "\eye" + "1" + str(i) + ".bmp")])

        savedCords.append([np.load('ReplacementEyes\eyeCords' + "0" + str(i) + '.npy'),
                           np.load('ReplacementEyes\eyeCords' + "1" + str(i) + '.npy')])

    currentEyeNumber = 0
    aboveScreen = np.load('ReplacementEyes\camerapos.npy')
    done = True
    preFrame = frame.copy()
    pre = None

    blink = 0
    if VirtualCameraOutput:
        cam = pyvirtualcam.Camera(width=WIDTH, height=HIGHT, fps=30)
    else:
        cam = None

    while done:

        val = intraFace.detect(frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        success = (val == 1)

        if success:
            pre, eyeCords = stabilizePoints(frame, preFrame, pre, X0, blink)
            if (not (detectBlink(eyeCords))):
                if blink > 0:
                    blink -= 1
                boudingbox(frame, eyeCords, 1)
                box = boudingbox(frame, eyeCords, 2)
                bigbox = boudingbox(frame, eyeCords, 4)

                translationRange = 7
                yawRange = 30
                rollRange = 40
                if not aboveScreen:
                    pitchUpper = 40
                    pitchLower = -20
                else:
                    pitchUpper = 20
                    pitchLower = -40
                if -yawRange < angle[0][1] < yawRange and -rollRange < angle[0][0] < rollRange and pitchLower < \
                        angle[0][2] < pitchUpper:
                    # Checking the negative edge of the frame
                    if (np.array(bigbox) > 0).all():
                        savedEye = savedEyes[currentEyeNumber]
                        savedCord = savedCords[currentEyeNumber]
                        frame2 = colourCorrectAndFrameBlend(frame, box, bigbox, eyeCords, savedEye, savedCord, angle[0],
                                                            aboveScreen)

                        alpha = 1
                        if -yawRange + translationRange > angle[0][1] or angle[0][1] > yawRange - translationRange:
                            z = (np.absolute(angle[0][1]) - (yawRange - translationRange)) / (
                                    yawRange - (yawRange - translationRange))
                            alpha = min(alpha, z)

                        if -rollRange + translationRange > angle[0][0] or angle[0][0] > rollRange - translationRange:
                            z = (np.absolute(angle[0][0]) - (rollRange - translationRange)) / (
                                    rollRange - (rollRange - translationRange))
                            alpha = min(alpha, z)

                        if pitchLower + translationRange > angle[0][2]:
                            z = (angle[0][2] - pitchLower) / ((pitchLower + translationRange) - pitchLower)
                            alpha = min(alpha, z)

                        if pitchUpper - translationRange < angle[0][2]:
                            z = (angle[0][2] - (pitchUpper - translationRange)) / (
                                    pitchUpper - (pitchUpper - translationRange))
                            alpha = min(alpha, z)

                        if alpha != 1:
                            alpha = np.round(alpha, 3)
                            frame2 = frame * (1 - alpha) + frame2 * alpha
                            frame2 = np.uint8(frame2)
                    else:
                        success = False
                else:
                    success = False
            else:
                blink = 8
                success = False
        if not success:
            currentEyeNumber = np.random.randint(0, 5)
            pre = None
            frame2 = frame.copy()

        if cam != None:
            if success:
                cam.send(frame2)
            else:
                cam.send(frame)
        else:
            cv2.imshow("Frame", frame)
            cv2.imshow("Frame2", frame2)


        k = cv2.waitKey(1)
        # excape
        if k == 27:
            break
        if k == 108:  # l swaping live eye
            currentEyeNumber = (currentEyeNumber + 1) % 5
        if k == 107:  # k swapping live eye
            currentEyeNumber = (currentEyeNumber - 1) % 5
        if k == 32:  # saving current eye as a new eye

            np.save('ReplacementEyes\eyeYAW' + str(currentEyeNumber) + '.npy', angle[0][1])

            for i in range(2):
                eyeCord = box[i]
                sbox = np.int0(eyeCord)
                biCord = bigbox[i]
                Bbox = np.int0(biCord)
                (Btopx, Btopy) = (np.min(Bbox[:, 0]), np.min(Bbox[:, 1]))
                (Bbotx, Bboty) = (np.max(Bbox[:, 0]), np.max(Bbox[:, 1]))
                Beye = (frame[Btopy:Bboty, Btopx:Bbotx]).copy()
                boundInB = sbox - [Btopx, Btopy]
                np.save('ReplacementEyes\eyeCords' + str(i) + str(currentEyeNumber) + '.npy', boundInB)
                cv2.imwrite(os.path.abspath("ReplacementEyes") + "\eye" + str(i) + str(currentEyeNumber) + ".bmp",
                            img=Beye)

            # Re-loading eye
            savedEyes[currentEyeNumber] = [
                cv2.imread(os.path.abspath("ReplacementEyes") + "\eye" + "0" + str(currentEyeNumber) + ".bmp"),
                cv2.imread(os.path.abspath("ReplacementEyes") + "\eye" + "1" + str(currentEyeNumber) + ".bmp")]
            savedCords[currentEyeNumber] = [np.load('ReplacementEyes\eyeCords' + "0" + str(currentEyeNumber) + '.npy'),
                                            np.load('ReplacementEyes\eyeCords' + "1" + str(currentEyeNumber) + '.npy')]

        preFrame = frame

        done, frame = cap.read()


def initiation(X0, angle, intraFace, ):
    _, frame = cap.read()
    while True:
        _, frame = cap.read()
        cv2.putText(frame, "Please press a or b to signal if ",
                    (int(0.05 * WIDTH), int(0.125 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=1)
        cv2.putText(frame, "the camera is located above or below the screen",
                    (int(0.05 * WIDTH), int(0.175 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=1)
        cv2.putText(frame, "above or below the screen",
                    (int(0.05 * WIDTH), int(0.225 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=1)
        cv2.putText(frame, "Press excape to skip", (int(0.05 * WIDTH), int(0.4 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1)
        cv2.putText(frame, "and use last values", (int(0.05 * WIDTH), int(0.455 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1)

        k = cv2.waitKey(5)
        cv2.imshow("Frame", frame)

        if k == 27:  # excape
            return

        if k == ord('a'):
            np.save('ReplacementEyes\camerapos.npy', True)
            break
        if k == ord('b'):
            np.save('ReplacementEyes\camerapos.npy', False)
            break

    pre = None
    currentEyeNumber = 0
    blink = 0

    while True:
        if currentEyeNumber == 5:
            return
        preFrame = frame.copy()
        _, frame = cap.read()
        val = intraFace.detect(frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)))
        success = (val == 1)
        frame2 = frame.copy()
        pitchangle = angle[0][2]
        cv2.putText(frame, "Press space while looking at the camera ",
                    (int(0.05 * WIDTH), int(0.125 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=1)
        cv2.putText(frame, "to save a image to replace with",
                    (int(0.05 * WIDTH), int(0.175 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 0), fontScale=1)
        cv2.putText(frame2, "Head Pitch " + str(round(pitchangle)) + "",
                    (int(0.02 * WIDTH), int(0.2 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), fontScale=1)
        cv2.putText(frame2, "Current Number " + str(currentEyeNumber) + "", (int(0.02 * WIDTH), int(0.3 * HIGHT)),
                    cv2.FONT_HERSHEY_SIMPLEX, color=(255, 0, 0), fontScale=1)

        k = cv2.waitKey(5)
        cv2.imshow("Frame", frame2)
        if success:
            pre, eyeCords = stabilizePoints(frame, preFrame, pre, X0, blink)
            if (not (detectBlink(eyeCords))):
                if blink > 0:
                    blink -= 1
                box = boudingbox(frame, eyeCords, 2)
                bigbox = boudingbox(frame, eyeCords, 4)
                if k == 27:  # excape
                    return
                if k == 32:  # space
                    cv2.imwrite(os.path.abspath("savedImages") + "\LiveEye" + ".jpg", img=frame)

                    for i in range(2):
                        eyeCord = box[i]
                        sbox = np.int0(eyeCord)
                        biCord = bigbox[i]
                        Bbox = np.int0(biCord)
                        (Btopx, Btopy) = (np.min(Bbox[:, 0]), np.min(Bbox[:, 1]))
                        (Bbotx, Bboty) = (np.max(Bbox[:, 0]), np.max(Bbox[:, 1]))
                        Beye = (frame[Btopy:Bboty, Btopx:Bbotx])
                        boundInB = sbox - [Btopx, Btopy]
                        np.save('ReplacementEyes\eyeCords' + str(i) + str(currentEyeNumber) + '.npy', boundInB)
                        cv2.imwrite(
                            os.path.abspath("ReplacementEyes") + "\eye" + str(i) + str(currentEyeNumber) + ".bmp",
                            img=Beye)
                    currentEyeNumber += 1


def detectBlink(eyesCords):
    for i in range(2):
        eyenp = np.array(eyesCords[i])
        v = eyenp[2] - eyenp[4]
        u = eyenp[1] - eyenp[5]
        hight = np.linalg.norm((u + v) / 2)
        width = np.linalg.norm(eyenp[3] - eyenp[0])
        if width / hight > 3.8:
            print("blinked: ", str(width / hight))
            return True
    return False


def colourCorrectAndFrameBlend(frame, boxcords, bigboxcords, eyeCords, savedEye, savedCords, angle, aboveScreen):
    frame = frame.copy()
    for i in range(2):
        boxcord = boxcords[i]
        box = boxcord.astype(np.int0)
        biCord = bigboxcords[i]
        Bbox = biCord.astype(np.int0)
        eyeCordsint = eyeCords[i].astype(np.int0)
        BsavedEye = savedEye[i]
        BsavedCords = savedCords[i]

        (Btopx, Btopy) = (np.min(Bbox[:, 0]), np.min(Bbox[:, 1]))
        (Bbotx, Bboty) = (np.max(Bbox[:, 0]), np.max(Bbox[:, 1]))


        # laplcisain can somewhat work with no colour correction
        # Say what type of blending to be used:
        # ff- full frame laplcisain,
        # el- just the eye laplcisain,
        # b- normal blur edges
        # n- no blending just overlaying with mask
        blendingType = "el"

        if blendingType == "pos":
            if (Btopx + Bbotx) % 2 == 1:
                Btopx -= 1

            if (Btopy + Bboty) % 2 == 1:
                Btopy -= 1

        Beye = (frame[Btopy:Bboty, Btopx:Bbotx]).copy()
        boundInB = box - [Btopx, Btopy]

        Twarp = time.perf_counter()
        vertAngle = angle[2]

        v = eyeCordsint[2] - eyeCordsint[4]
        u = eyeCordsint[1] - eyeCordsint[5]
        vert = (u + v) / 4
        hor = (eyeCordsint[0] - eyeCordsint[3]) / 2

        if aboveScreen == False:
            # In this case we are looking up hense eyes are slighly to big.
            # Only working for angles -10 to 30
            # Getting z in the range -1 to 1
            z = ((vertAngle + 10) / 40 - 0.5) * 2
            if i == 0:
                boundInB[0] = boundInB[0] - vert / 5 + vert * z / 6
                boundInB[1] = boundInB[1] - vert / 4 + vert * z / 6
                warp_mat = cv2.getAffineTransform(BsavedCords[:3].astype(np.float32), boundInB[:3].astype(np.float32))
            else:
                boundInB[0] = boundInB[0] - vert / 4 + vert * z / 6
                boundInB[1] = boundInB[1] - vert / 5 + vert * z / 6
                BsavedCords[2] = BsavedCords[3]
                boundInB[2] = boundInB[3]
                warp_mat = cv2.getAffineTransform(BsavedCords[:3].astype(np.float32), boundInB[:3].astype(np.float32))
        else:
            # Only working for angles -30 to 10
            # Getting z in the range -1 to 1 again
            z = ((vertAngle + 30) / 40 - 0.5) * 2
            if i == 0:
                boundInB[0] = boundInB[0] + vert / 5 - vert * z / 6
                boundInB[1] = boundInB[1] + vert / 4 - vert * z / 6
                warp_mat = cv2.getAffineTransform(BsavedCords[:3].astype(np.float32), boundInB[:3].astype(np.float32))
            else:
                boundInB[0] = boundInB[0] + vert / 4 - vert * z / 6
                boundInB[1] = boundInB[1] + vert / 5 - vert * z / 6
                BsavedCords[2] = BsavedCords[3]
                boundInB[2] = boundInB[3]
                warp_mat = cv2.getAffineTransform(BsavedCords[:3].astype(np.float32), boundInB[:3].astype(np.float32))

        Bfittedeye = cv2.warpAffine(BsavedEye, warp_mat, (Beye.shape[1], Beye.shape[0]),
                                    borderMode=cv2.BORDER_REPLICATE)

        # Try Perspective transform
        # warp_mat = cv2.getPerspectiveTransform(BsavedCords.astype(np.float32), boundInB.astype(np.float32))
        # Bfittedeye = cv2.warpPerspective(BsavedEye, warp_mat, (Beye.shape[1], Beye.shape[0]),borderMode=cv2.BORDER_REPLICATE)

        # Making predicted eyes:


        if i == 0:
            eyeCordsint[4] -= (0.3 * vert).astype(dtype=np.int)
            eyeCordsint[5] -= (0.3 * vert).astype(dtype=np.int)
            eyeCordsint[0] -= (0.2 * vert).astype(dtype=np.int) + (0.2 * hor).astype(dtype=np.int)
            eyeCordsint[1] -= (0.3 * vert).astype(dtype=np.int)
            eyeCordsint[2] -= (0.3 * vert).astype(dtype=np.int)
        else:
            eyeCordsint[4] -= (0.3 * vert).astype(dtype=np.int)
            eyeCordsint[5] -= (0.3 * vert).astype(dtype=np.int)
            eyeCordsint[3] -= (0.2 * vert).astype(dtype=np.int) - (0.2 * hor).astype(dtype=np.int)
            eyeCordsint[1] -= (0.3 * vert).astype(dtype=np.int)
            eyeCordsint[2] -= (0.3 * vert).astype(dtype=np.int)

        BeyeMask = np.zeros_like(Beye[:, :, 1])
        BeyeMask = cv2.ellipse(BeyeMask, cv2.fitEllipse(eyeCordsint - [Btopx, Btopy]), color=255, thickness=-1)


        bpredeyes1 = np.empty_like(Beye)
        bpredeyes2 = np.empty_like(Beye)

        for c in range(3):
            bpredeyes1[:, :, c] = fitEye(Beye[:, :, c])
            bpredeyes2[:, :, c] = fitEye(Bfittedeye[:, :, c])

        div = np.divide(bpredeyes1, bpredeyes2)

        Corrected = Bfittedeye * div

        bcolourcorrected = np.empty_like(Bfittedeye)

        k = int(int(np.linalg.norm(hor) / 3) * 2 + 1)
        blur = cv2.GaussianBlur(BeyeMask, (k, k), 0)

        for c in range(3):
            # bcolourcorrected[:, :, c] = Bfittedeye[:, :, c] * (blur / 255) + (1 - blur / 255) * Corrected[:, :, c]
            bcolourcorrected[:, :, c] = Bfittedeye[:, :, c] * (BeyeMask / 255) + (1 - BeyeMask / 255) * Corrected[:, :,
                                                                                                        c]
        # bcolourcorrected = Corrected

        bound(bcolourcorrected)

        bcolourcorrected = bcolourcorrected.astype(np.uint8)


        # 1)  A tight oval only around eye centre
        # 2)  A oval covering eye and eyelid
        # 3)  A large ovel covering all to the eyelid
        # 4)  A tight retangle coving eye section

        maskType = 2

        if maskType == 1:
            innermask = BeyeMask

        if maskType == 2:
            BeyeMask = cv2.ellipse(BeyeMask, cv2.fitEllipse(eyeCordsint - [Btopx, Btopy]), color=255,
                                   thickness=int(np.linalg.norm(vert * 2)))
            innermask = BeyeMask

        if maskType == 3:
            BeyeMask = cv2.ellipse(BeyeMask, cv2.fitEllipse(eyeCordsint - [Btopx, Btopy]), color=255,
                                   thickness=int(np.linalg.norm(vert * 5)))
            innermask = BeyeMask

        if maskType == 4:
            fmask = np.zeros_like(BeyeMask)
            cv2.drawContours(fmask, [box - [Btopx, Btopy]], 0, 255, -1)  # Draw filled contour in mask
            cv2.drawContours(fmask, [box - [Btopx, Btopy]], 0, 0,
                             int(np.linalg.norm(hor * 0.5)))  # Draw filled contour in mask
            innermask = fmask

        Tblending = time.perf_counter()

        if blendingType == "pos":
            centre = (Btopx + (Bbotx - Btopx) / 2, Btopy + (Bboty - Btopy) / 2)
            centre = (int(centre[0]), int(centre[1]))

            output = cv2.seamlessClone(bcolourcorrected, frame, innermask, centre, cv2.NORMAL_CLONE)
            frame = output


        if blendingType == "ff":
            fmask = np.zeros_like(frame[:, :, 0])
            fmask[Btopy:Bboty, Btopx:Bbotx] = innermask

            ceye = frame.copy()
            # Only replacing the eyes
            ceye[Btopy:Bboty, Btopx:Bbotx] = bcolourcorrected
            depth = int(np.log2(np.linalg.norm(4 * hor)))

            frame = laplacianSameSize(frame, ceye, fmask, depth)


        if blendingType == "el":
            depth = int(np.log2(np.linalg.norm(2 * hor)))

            beyeblend = laplacianSameSize(Beye, bcolourcorrected, innermask, depth)
            frame[Btopy:Bboty, Btopx:Bbotx] = beyeblend


        if blendingType == "combo":
            depth = int(np.log2(np.linalg.norm(2 * hor)))
            beyeblend = laplacianSameSize(Beye, bcolourcorrected, innermask, depth)

            k = int(int(np.linalg.norm(hor) / 4) * 2 + 1)
            blur = cv2.GaussianBlur(innermask, (k, k), k)

            for c in range(3):
                frame[Btopy:Bboty, Btopx:Bbotx, c] = (
                        frame[Btopy:Bboty, Btopx:Bbotx, c] * (1 - blur / 255) + beyeblend[:, :, c] * (
                        blur / 255)).astype(np.uint8)


        if blendingType == "blur":
            k = int(int(np.linalg.norm(hor) / 3) * 2 + 1)
            blur = cv2.GaussianBlur(innermask, (k, k), 0)

            doublesizeshow("blur", blur)

            for c in range(3):
                frame[Btopy:Bboty, Btopx:Bbotx, c] = (
                        frame[Btopy:Bboty, Btopx:Bbotx, c] * (1 - blur / 255) + bcolourcorrected[:, :, c] * (
                        blur / 255)).astype(np.uint8)

        if blendingType == "n":
            frame[Btopy:Bboty, Btopx:Bbotx][innermask == 255] = bcolourcorrected[innermask == 255]

        if blendingType == "full":
            frame[Btopy:Bboty, Btopx:Bbotx] = bcolourcorrected


    return frame


def gaussianPyramid(img, num_levels):
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
    # Appling the mask
    for lFrame, lCEye, gMask in zip(lpFrame, lpCEye, gpMask):
        lFrame[gMask == 255] = lCEye[gMask == 255]
        LS.append(lFrame)

    # now reconstruct
    ls_ = LS[0]
    for i in range(1, levels + 1):
        size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize=size)
        ls_ = ls_ + LS[i]

    # Making it above 0 before becoming uint8
    bound(ls_)
    return ls_.astype(np.uint8)


# Until functions
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

    Z = eye.ravel()
    OnesX = np.ones(eye.shape[0])
    OnesY = np.ones(eye.shape[1])
    XY = np.outer(X, Y).ravel()
    Y2 = Y ** 2
    X2 = X ** 2

    A = np.array([np.ones_like(XY), np.outer(OnesX, Y).ravel(), np.outer(OnesX, Y2).ravel(),
                  np.outer(OnesX, Y2 * Y).ravel(), np.outer(X, OnesY).ravel(), XY,
                  np.outer(X, Y2).ravel(), np.outer(X2, OnesY).ravel(), np.outer(X2, Y).ravel(),
                  np.outer(X2 * X, OnesY).ravel()]).T

    x, _, _, _ = scipy.linalg.lstsq(A, Z, lapack_driver='gelsy', overwrite_b=True, check_finite=False)

    minSolution = A * x
    neweye = minSolution.sum(axis=1)


    neweye[neweye < 8] = 8
    neweye[neweye > 255] = 255

    neweye = neweye.reshape((X.shape[0], Y.shape[0]))
    return neweye


def boudingbox(frame, eyeCords, scale):
    eyes = []
    for i in range(2):
        box = np.zeros(shape=(4, 2))
        eyenp = np.array(eyeCords[i])
        v = eyenp[2] - eyenp[4]
        u = eyenp[1] - eyenp[5]
        vert = (u + v) / 4
        vert = vert * scale
        hor = (eyenp[3] - eyenp[0]) / 2
        hord = (hor * scale - hor) / 2
        box[0] = eyenp[0] + vert - hord
        box[3] = eyenp[0] - vert - hord
        box[1] = eyenp[3] + vert + hord
        box[2] = eyenp[3] - vert + hord

        intBox = np.int0(box)
        draw = False
        if draw:
            if scale == 1:
                cv2.drawContours(frame, [intBox], 0, (0, 0, 0), 2)
            else:
                cv2.drawContours(frame, [intBox], 0, (0, 0, 255), 2)
        eyes.append(box)
    return eyes


def stabilizePoints(frame, preFrame, pre, points, blink, ):
    points = points.reshape((49 * 2), order='F')
    points = points.reshape((49, 2), order='A')

    draw = False
    if draw:
        i = 0
        for point in points:
            x = point[0]
            y = point[1]
            cv2.circle(frame, (int(float(x)), int(float(y))), 2, (255, 255, 0), -1)
            i += 1

    points = points[19:31]
    hor = (np.linalg.norm(points[0] - points[3] + points[6] - points[9])) / 4
    s = int(hor * 2.5)
    maxLevel = int(hor / 3)

    lk_params = dict(winSize=(s, s), maxLevel=maxLevel,
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 12, 0.03))

    # If not detected properly last time do not stablise
    if pre is None or blink > 0:
        return points, [points[0:6], points[6:12]]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    graypre = cv2.cvtColor(preFrame, cv2.COLOR_BGR2GRAY)

    OFpoints, status, err = cv2.calcOpticalFlowPyrLK(graypre, gray, pre, None, **lk_params)

    for i in range(0, len(points)):
        d = np.linalg.norm(points[i] - pre[i] + 5)
        alpha = math.exp(-d * d / 100)
        OFpoints[i] = (1 - alpha) * points[i] + alpha * OFpoints[i]

    dif = pre - OFpoints
    dif = dif.mean(axis=0)
    if abs(dif[0]) < 0.4 and abs(dif[1]) < 0.4:
        stablePoints = pre * 7 / 8 + OFpoints / 8
    elif abs(dif[0]) < 1.5 and abs(dif[1]) < 1.5:
        stablePoints = pre / 4 + OFpoints * 3 / 4
    else:
        stablePoints = OFpoints

    return stablePoints, [stablePoints[0:6], stablePoints[6:12]]


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
            if 11 <= i < 15:
                noseCords.append((x, y))
    return eyeCords, noseCords


def loaddll():
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_core246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_ffmpeg246_64.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_flann246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_highgui246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_imgproc246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_objdetect246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_features2d246.dll'))
    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\opencv_calib3d246.dll'))

    ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\IntraFaceDLL.dll'))
    # intraFace = ctypes.cdll.LoadLibrary(os.path.abspath('IntraFaceResources\\IntraFaceTracker.dll'))
    intraFace = ctypes.cdll.LoadLibrary(
        'C:\\Users\\Marcus\\OneDrive\\Documents\\GitHub\\IntraFace\\x64\\Release\\IntraFaceTracker.dll')

    return intraFace


if __name__ == "__main__":
    run()
