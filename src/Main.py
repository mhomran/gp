from PlayerDetection.PlayerDetection import PlayerDetction
from PlayerDetection.ImageClass import ImageClass
from ModelField.model_field import ModelField
import numpy as np
import pickle
import cv2 as cv
import copy
import timeit
IMG = ImageClass('../data/videos/Video.avi')

# ret, frame, frameId = IMG.readFrame()
# MF = ModelField(frame)
# particles = MF._get_particles()

# with open('file2.pkl', 'wb') as f:
#     pickle.dump(particles, f)
# f.close()

particles_ORG = {}
with open('file.pkl', 'rb') as f:
    particles_ORG = pickle.load(f)

particles = copy.deepcopy(particles_ORG)
PD = PlayerDetction(particles, IMG)

while True:

    ret, frame, frameId = IMG.readFrame()

    if frame is None:
        break

    fgMask = PD.subBG(frame)
    IMG.writeTxt(frame)
    IMG.showImage(frame, "Frame")

    if(frameId > 300):
        PD.preProcessing(fgMask)
        PD.loopOnBB()

    wait = 1
    if(frameId > 300):
        wait = 0
    keyboard = cv.waitKey(wait)
    if keyboard == 'q' or keyboard == 27:
        break


# save model
# with open('file.pkl', 'wb') as f:
#     pickle.dump(particles, f)
# f.close()
