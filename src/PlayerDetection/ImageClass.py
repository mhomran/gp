import cv2 as cv


class ImageClass:
    def __init__(self, path):
        self.cap = cv.VideoCapture(path)

    def readFrame(self):
        ret, frame = self.cap.read()
        frame = self.resizeImage(frame)
        frameId = int(self.cap.get(1))
        return ret, frame, frameId

    def showImage(self, frame, windowName, resize=(960, 720)):
        img = cv.resize(frame, resize)
        cv.imshow(windowName, img)

    def writeImage(self, frame, imgName):
        cv.imwrite(imgName, frame)

    def resizeImage(self, frame, resize=(960, 720)):
        img = cv.resize(frame, resize)
        return img

    def writeTxt(self, frame):
        cv.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
        cv.putText(frame, str(self.cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
