import cv2 as cv
import imutils

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
MFfrmae = cv.imread('./people.jpg')
img = imutils.resize(MFfrmae, 1200)

boundingBoxes, wrights = hog.detectMultiScale(
    img, winStride=(1, 1), padding=(8, 8), scale=1)


for (x, y, w, h) in boundingBoxes:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

cv.imshow('people', img)
cv.waitKey(0)
