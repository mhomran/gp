import cv2 as cv
import imutils


class ImageClass:
    def __init__(self, path=None):
        if path is not None:
            self.cap = cv.VideoCapture(path)

    def readFrame(self):
        ret, frame = self.cap.read()
        frame = self.resizeImage(frame)
        frameId = int(self.cap.get(1))
        return ret, frame, frameId

    def showImage(self, frame, windowName, width=1200):
        img = imutils.resize(frame, width)
        cv.imshow(windowName, img)

    def writeImage(self, frame, imgName):
        cv.imwrite(imgName, frame)

    def resizeImage(self, frame, resize=(960, 720)):
        img = imutils.resize(frame, 1200)
        return img

    def writeTxt(self, frame, id):
        cv.rectangle(frame, (30, 30), (350, 130), (255, 255, 255), -1)
        cv.putText(frame, str(id), (120,120),
                   cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2)
