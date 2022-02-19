import cv2 as cv
import numpy as np

BACK_SUB_HISTORY = 500
BACK_SUB_THRE = 16
BACK_SUB_DETECT_SHADOW = False

KERNEL_SIZE = (2, 2)


class BoundingBox:
    def __init__(self, tl, br):
        self.tl = tl
        self.br = br


class PlayerDetction:
    def __init__(self, particles, IMG):
        self.backSub = cv.createBackgroundSubtractorMOG2(
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

    def subBG(self, frame):
        self.frame = frame
        # will be removed
        self.setFramesForDisplay()
        frame = self.backSub.apply(frame)
        # shadow
        #_, frame = cv.threshold(frame, 254, 255, cv.THRESH_BINARY)
        return frame

    def setFramesForDisplay(self):
        self.MFfrmae = self.frame.copy()
        self.MFBefore = self.frame.copy()
        self.contourFrame = self.frame.copy()

    def preProcessing(self, fgMask):
        self.opening(fgMask)
        self.setContours()
        self.setBoundingBoxes()

        self.IMG.showImage(self.contourFrame, "contours")

        self.fgMask = fgMask

    def opening(self, frame):
        self.openingImg = cv.morphologyEx(frame, cv.MORPH_OPEN, self.kernel)

    def setContours(self):
        edged = cv.Canny(self.openingImg, 30, 200)
        contours, _ = cv.findContours(edged,
                                      cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours

    def setBoundingBoxes(self):
        BB = []
        for c in self.contours:
            rect = cv.boundingRect(c)
            x, y, w, h = rect
            if(w*h < 20):
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

    def getRoi(self, B):
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

    def getRatio(self, img):
        number_of_white_pix = np.sum(img == 255)
        number_of_black_pix = np.sum(img == 0)
        percentage = round(number_of_white_pix /
                           (number_of_white_pix+number_of_black_pix), 2)

        return percentage

    def getDecision(self, p1, p2, p3):
        if(p1 < .15):
            return False, 0
        if(p2 < .15):
            return False, 0
        if(p3 < .15):
            return False, 0

        return True, round((p1+p2+p3)/3, 2)

    def IOU(self, B1, B2):
        dx = min(B1.br[0], B2.br[0]) - max(B1.tl[0], B2.tl[0])
        dy = min(B1.br[1], B2.br[1]) - max(B1.tl[1], B2.tl[1])

        area1 = (B1.br[0]-B1.tl[0])*(B1.br[1]-B1.tl[1])
        area2 = (B2.br[0]-B2.tl[0])*(B2.br[1]-B2.tl[1])

        intersection = dx*dy
        union = area1+area2-intersection

        assert union != 0
        return round((intersection/union), 2)

    def applyNonMax(self, particles):
        i = 0
        while(i < len(particles)):
            j = i+1
            while(j < len(particles)):
                iou = self.IOU(particles[i].B, particles[j].B)
                if(iou > .2):
                    particles.pop(j)
                    j = j-1
                j = j+1
            i = i+1
        return particles

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
        roi1, roi2, roi3 = self.getRoi(MFBB[particle].B)
        pRo1 = self.getRatio(self.fgMask[roi1[0]:roi1[1], roi1[2]:roi1[3]])
        pRo2 = self.getRatio(self.fgMask[roi2[0]:roi2[1], roi2[2]:roi2[3]])
        pRo3 = self.getRatio(self.fgMask[roi3[0]:roi3[1], roi3[2]:roi3[3]])

        decision, ratio = self.getDecision(pRo1, pRo2, pRo3)

        if(decision):
            MFBB[particle].ratio = ratio
            NonMax.append(MFBB[particle])

    def loopOnBB(self):
        for _, B in enumerate(self.BB):
            MFBB = self.getParticlesInBB(B)
            # draw BB
            cv.rectangle(self.frame, B.tl,
                         B.br, (0, 255, 0), 1)
            self.IMG.showImage(self.frame, "BB")

            # get candidate particle
            NonMax = []
            for particle in list(MFBB):
                self.getCandidateParticle(MFBB, particle,  NonMax)

            for particle in NonMax:
                cv.rectangle(self.MFBefore, particle.B.tl,
                             particle.B.br, (255, 0, 0), 1)

            self.IMG.showImage(self.MFBefore, "MFBeforeNonMax")

            # apply nonmax supression
            NonMax.sort(key=lambda x: x.ratio, reverse=True)
            NonMax = self.applyNonMax(NonMax)

            # draw particles
            for particle in NonMax:
                cv.rectangle(self.MFfrmae, particle.B.tl,
                             particle.B.br, (255, 0, 0), 1)

            self.IMG.showImage(self.MFfrmae, "MFBB After non max")
            keyboard = cv.waitKey(0)
            if keyboard == 'q' or keyboard == 27:
                break
