import random
import cv2
import numpy as np
from numpy.lib.function_base import _cov_dispatcher
from classifier import *
import imutils
from final import *

videoName = input("Enter the video name: ")
path = "C://Users//thilak//Desktop//Major-Project//Violenceproj//Final_Test//" + \
    videoName + ".mp4"
cap = cv2.VideoCapture(path)

width = int(cap.get(3))
height = int(cap.get(4))

acc = "ACCURACY: " + str(accuracy(path)) + "%"
color, tag = cs(path)

if (cap.isOpened() == False):
    print("Error opening video stream or file")
while (cap.isOpened()):
    ret, frame = cap.read()
    try:
        frame = imutils.resize(frame, width=600)
    except(AttributeError):
        pass
    font = cv2.FONT_HERSHEY_DUPLEX
    x, y, w, h = 5, 5, 270, 80
    cv2.rectangle(frame, (x, x), (x + w, y + h), (255, 255, 255), -1)
    cv2.putText(frame, tag, (20, 35), font, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, acc, (20, 70), font, 0.8, (0, 0, 0), 2)
    frame = cv2.copyMakeBorder(frame,
                               100,
                               100,
                               100,
                               100,
                               cv2.BORDER_CONSTANT,
                               value=color)
    if ret == True:
        cv2.imshow('Frame', frame)
        if cv2.waitKey(35) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
