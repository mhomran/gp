from PlayerDetection.PlayerDetection import PlayerDetection
from PlayerDetection.ImageClass import ImageClass
from ModelField.model_field import ModelField
import numpy as np
import pickle
import cv2 as cv
import copy
import timeit
import imutils
IMG = ImageClass('../data/videos/Video.avi')

BGIMG = cv.imread('./backGround.png')

BGIMG = imutils.resize(BGIMG, 1200)

# mf_gui_clicks = [(311, 110), (616, 101), (922, 103),  # the three top corners
#                  (1196, 223), (619, 265), (27, 240),  # the three bottom corners
#                  (195, 162), (193, 142),  # the left post corners
#                  (1035, 152), (1037, 132)]  # the right post corners

# ret, frame, frameId = IMG.readFrame()
# MF = ModelField(frame, 3, mf_gui_clicks)


# with open('MF.pkl', 'wb') as f:
#     pickle.dump(MF, f)
# f.close()

MF = {}
with open('MF.pkl', 'rb') as f:
    MF = pickle.load(f)

PD = PlayerDetection(MF, IMG, BGIMG)


while True:

    ret, frame, frameId = IMG.readFrame()

    if frame is None:
        break

    fgMask = PD.subBG(frame, frameId)
    IMG.writeTxt(frame)
    IMG.showImage(frame, "Frame")

    PD.preProcessing(fgMask)
    PD.loopOnBB()
    cv.waitKey(1)

    keyboard = cv.waitKey(0)
    if keyboard == 'q' or keyboard == 27:
        break
