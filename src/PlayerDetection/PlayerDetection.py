import cv2 as cv
import numpy as np
import csv
from imutils.object_detection import non_max_suppression

import time

BACK_SUB_HISTORY = 500
BACK_SUB_THRE = 16
BACK_SUB_DETECT_SHADOW = False

KERNEL_SIZE = (2, 2)
MIN_WIDTH_BB = 20

VERTICAL_TH = .15
HORIZONTAL_TH = .2
IOU_TH = .2


class BoundingBox:
    def __init__(self, tl, br):
        self.tl = tl
        self.br = br


class PlayerDetection:
    LOW_TH = 30
    HIGH_TH = 200
    LEARNING_RATE = -1

    def __init__(self, particles, IMG):
        self.backSub = cv.cuda.createBackgroundSubtractorMOG2(
            BACK_SUB_HISTORY, BACK_SUB_THRE, BACK_SUB_DETECT_SHADOW)

        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, KERNEL_SIZE)
        self.contours = []
        self.particles = particles
        self.openingImg = None
        self.fgMask = None
        self.BB = None

        self.IMG = IMG
        self.frame = None
        self.MFfrmae = None
        self.MFBefore = None
        self.contourFrame = None
        self.outputPD = None

        self.morph_filter = cv.cuda.createMorphologyFilter(cv.MORPH_OPEN, cv.CV_8UC1, self.kernel)
        self.canny_filter = cv.cuda.createCannyEdgeDetector(PlayerDetection.LOW_TH, PlayerDetection.HIGH_TH)

        self.frame_gpu = cv.cuda_GpuMat()

    def subBG(self, frame_gpu):
        self.frame = frame_gpu.download()
        # will be removed
        self.setFramesForDisplay()
        fgMask_gpu = self.backSub.apply(frame_gpu, PlayerDetection.LEARNING_RATE, None)
        return fgMask_gpu.download()

    def setFramesForDisplay(self):
        self.MFfrmae = self.frame.copy()
        self.MFBefore = self.frame.copy()
        self.contourFrame = self.frame.copy()

    def preProcessing(self, fgMask):
        self.fgMask = fgMask

        self.frame_gpu.upload(fgMask)

        self.openingImg = self.morph_filter.apply(self.frame_gpu)
        edged = self.canny_filter.detect(self.openingImg).download()
        self.contours, _ = cv.findContours(edged,
                                      cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.setBoundingBoxes()


    def setBoundingBoxes(self):
        BB = []
        for c in self.contours:
            rect = cv.boundingRect(c)
            x, y, w, h = rect
            if(w*h < MIN_WIDTH_BB or w > h):
                continue

            tl = (x, y)
            br = (x+w, y+h)
            B = BoundingBox(tl, br)
            BB.append(B)

            cv.rectangle(self.contourFrame, tl, br, (0, 0, 255), 1)
            cv.circle(self.contourFrame, tl, radius=1,
                      color=(255, 0, 0), thickness=-1)
        self.BB = BB

    def getBoundingBoxes(self):
        return self.BB

    def vertRoi(self, B):
        start_X = B.tl[0]
        start_Y = B.tl[1]
        end_X = B.br[0]
        end_Y = B.br[1]
        height = end_Y-start_Y
        inc = height//3
        roi1 = (start_Y, start_Y+inc, start_X, end_X)
        roi2 = (start_Y+inc, start_Y+2*inc, start_X, end_X)
        roi3 = (start_Y+2*inc, end_Y, start_X, end_X)
        return roi1, roi2, roi3

    def horiRoi(self, B):
        start_X = B.tl[0]
        start_Y = B.tl[1]
        end_X = B.br[0]
        end_Y = B.br[1]
        width = end_X-start_X
        middleX = end_X-width//2
        roiL = (start_Y, end_Y, start_X, middleX)
        roiR = (start_Y, end_Y, middleX, end_X)
        return roiL, roiR

    def getRatio(self, img):
        number_of_white_pix = np.sum(img == 255)
        number_of_black_pix = np.sum(img == 0)
        percentage = round(number_of_white_pix /
                           (number_of_white_pix+number_of_black_pix), 2)

        return percentage

    def getDecision(self, p1, p2, p3):
        if(p1 < VERTICAL_TH or p2 < VERTICAL_TH or p3 < VERTICAL_TH):
            return False, 0

        return True, round((p1+p2+p3)/3, 2)

    def getParticlesInBB(self, B):
        MFBB = {}
        Sx = B.tl[0]
        Ex = B.br[0]

        Sy = B.tl[1]
        Ey = B.br[1]

        # get all particle in BB
        for y in range(Sy, Ey+1):
            for x in range(Sx, Ex+1):
                if self.particles.get((x, y)):
                    MFBB[(x, y)] = self.particles[(x, y)]

        return MFBB

    def getCandidateParticle(self, MFBB, particle,  NonMax):
        roi1, roi2, roi3 = self.vertRoi(MFBB[particle].B)
        pRo1 = self.getRatio(self.fgMask[roi1[0]:roi1[1], roi1[2]:roi1[3]])
        pRo2 = self.getRatio(self.fgMask[roi2[0]:roi2[1], roi2[2]:roi2[3]])
        pRo3 = self.getRatio(self.fgMask[roi3[0]:roi3[1], roi3[2]:roi3[3]])
        decisionV, ratio = self.getDecision(pRo1, pRo2, pRo3)

        decisionH, pL, pR = self.hoizontalCheck(MFBB[particle])

        if(decisionV and decisionH):
            MFBB[particle].ratio = ratio
            NonMax.append(MFBB[particle])

    def hoizontalCheck(self, particle):
        roiL, roiR = self.horiRoi(particle.B)
        pL = self.getRatio(
            self.fgMask[roiL[0]:roiL[1], roiL[2]:roiL[3]])
        pR = self.getRatio(
            self.fgMask[roiR[0]:roiR[1], roiR[2]:roiR[3]])

        return (pL > HORIZONTAL_TH and pR > HORIZONTAL_TH), pL, pR

    def loopOnBB(self):
        candid = []
        for _, B in enumerate(self.BB):
            MFBB = self.getParticlesInBB(B)
            # draw BB
            cv.rectangle(self.frame, B.tl,
                         B.br, (0, 255, 0), 1)

            # get candidate particle
            NonMax = []
            for particle in list(MFBB):
                self.getCandidateParticle(MFBB, particle, NonMax)
            candid = candid + NonMax

        self.IMG.showImage(self.frame, "BB")

        rects = np.array([[particle.B.tl[0], particle.B.tl[1],
                         particle.B.br[0], particle.B.br[1]] for particle in candid])

        non_max = non_max_suppression(rects, probs=None, overlapThresh=IOU_TH)
        for (x1, y1, x2, y2) in non_max:
            cv.rectangle(self.MFfrmae, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        self.IMG.showImage(self.MFfrmae, "MFBB After non max")

        self.outputPD = non_max

    def getOutputPD(self):
        return self.outputPD
