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

# ret, frame, frameId = IMG.readFrame()
# MF = ModelField(frame)
# particles = MF._get_particles()

# with open('file3.pkl', 'wb') as f:
#     pickle.dump(particles, f)
# f.close()

particles_ORG = {}
with open('file3.pkl', 'rb') as f:
    particles_ORG = pickle.load(f)

particles = copy.deepcopy(particles_ORG)
PD = PlayerDetection(particles, IMG,BGIMG)


while True:

    ret, frame, frameId = IMG.readFrame()

    if frame is None:
        break

    fgMask = PD.subBG(frame)
    IMG.writeTxt(frame)
    IMG.showImage(frame, "Frame")

    
    PD.preProcessing(fgMask)
    PD.loopOnBB()
    cv.waitKey(1)

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
