import cv2 as cv

BACK_SUB_HISTORY = 500
BACK_SUB_THRE = 16
BACK_SUB_DETECT_SHADOW = False

KERNEL_SIZE = (2, 2)


class BoundingBox:
    def __init__(self, tl, br):
        self.tl = tl
        self.br = br


class PlayerDetction:
    def __init__(self):
        self.backSub = cv.createBackgroundSubtractorMOG2(
            BACK_SUB_HISTORY, BACK_SUB_THRE, BACK_SUB_DETECT_SHADOW)

        self.kernel = cv.getStructuringElement(cv.MORPH_RECT, KERNEL_SIZE)
        self.contours = []

    def subBG(self, frame):
        frame = self.backSub.apply(frame)
        return frame

    def opening(self, frame):
        openingImg = cv.morphologyEx(frame, cv.MORPH_OPEN, self.kernel)
        return openingImg

    def getContours(self, frame):
        edged = cv.Canny(frame, 30, 200)
        contours, _ = cv.findContours(edged,
                                      cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        return contours

    def drawContours(self, frame):
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

            cv.rectangle(frame, tl, br, (0, 0, 255), 1)
            cv.circle(frame, tl, radius=1, color=(255, 0, 0), thickness=-1)
        self.BB = BB

        return frame

    def getBoundingBoxes(self):
        return self.BB
