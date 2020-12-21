import sys, string, os, subprocess
import cv2

import ctypes
import numpy


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# os.chdir('C:\\Users\\Marcus\\OneDrive\\Documents\\GitHub\\IntraFace\\x64')
#
# os.system('C:\\Users\\Marcus\\OneDrive\\Documents\\GitHub\\IntraFace\\x64\\Release\\IntraFaceTracker.exe')

_, frame = cap.read()
print(frame.shape)
print(frame)


c_float_p = ctypes.POINTER(ctypes.c_int8)
data = frame
data = data.astype(numpy.intc)
data_p = data.ctypes.data_as(c_float_p)

print(data_p)



# p = subprocess.Popen(['C:\\Users\\Marcus\\OneDrive\\Documents\\GitHub\\IntraFace\\x64\\Release\\IntraFaceTracker.exe'],cwd='C:\\Users\\Marcus\\OneDrive\\Documents\\GitHub\\IntraFace\\x64'
#                     , stdout=subprocess.PIPE, stdin=subprocess.PIPE, shell=True)
#
# while True:
#     ret = p.stdout.readline().strip()
#     print(ret)
#
#     # _, frame = cap.read()
#     # print(frame.shape)
#     # frame = frame[0][0][:]
#     # print(frame)
#
#
#     value = str(10) + '\n'
#     value = bytes(value, 'UTF-8')  # Needed in Python 3.
#     p.stdin.write(value)
#     p.stdin.flush()
#
#
#     # p.stdin.write(bytes(10))
#     # # p.stdin.flush()
#     # print(p.stdout.read())
#
# # x = 0
# # while True:
# #     x+=1
# #     # subprocess.check_output()
# #     print(p.poll())