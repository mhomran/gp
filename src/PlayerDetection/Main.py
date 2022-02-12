from matplotlib.pyplot import draw
from ImageClass import *
from PlayerDetection import *
import cv2 as cv


IMG = ImageClass('../../../output.avi')
PD = PlayerDetction()
while True:

    ret, frame, frameId = IMG.readFrame()

    fgMask = PD.subBG(frame)
    IMG.writeTxt(frame)
    IMG.showImage(frame, "Frame")

    if(frameId > 300):
        openingImg = PD.opening(fgMask)
        PD.getContours(openingImg)
        PD.drawContours(frame)
        IMG.showImage(frame, "contours")

    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break
