import cv2 as cv

BACK_SUB_HISTORY = 500
BACK_SUB_THRE = 16
BACK_SUB_DETECT_SHADOW = False

KERNEL_SIZE = (2, 2)


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
                                      cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        self.contours = contours
        return contours

    def drawContours(self, frame):
        for c in self.contours:
            rect = cv.boundingRect(c)
            x, y, w, h = rect
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        return frame
