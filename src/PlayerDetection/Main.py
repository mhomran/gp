import sys
# setting path
sys.path.append('../')

from ModelField.model_field import ModelField
from PlayerDetection import *
from ImageClass import ImageClass
import pickle
import cv2 as cv
import copy

IMG = ImageClass('/media/mhomran/Material/Workspaces/GitHub/gp/data/videos/cuda_output.avi')

ret, frame, frameId = IMG.readFrame()
MF = ModelField(frame)
particles = MF._get_particles()

PD = PlayerDetection(particles, IMG)

frame_gpu = cv.cuda_GpuMat()

while True:

    ret, frame, frameId = IMG.readFrame()
    
    if frame is None:
        break

    frame_gpu.upload(frame)
    
    fgMask = PD.subBG(frame_gpu)
    IMG.writeTxt(frame, frameId)
    IMG.showImage(frame, "Frame")

    if(frameId > 300):
        cv.imshow("mask", fgMask)
        PD.preProcessing(fgMask)
        PD.loopOnBB()

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
