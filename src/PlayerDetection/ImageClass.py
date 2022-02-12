import cv2 as cv
import imutils


class ImageClass:
    def __init__(self, path):
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

    def writeTxt(self, frame):
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(self.cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
