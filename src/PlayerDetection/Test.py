import cv2 as cv
import imutils

cap = cv.VideoCapture('../data/videos/Video.avi')

Background = cv.imread('./backGround.png')


while True:
    ret, frame = cap.read()
    frameId = int(cap.get(1))
    if frame is None:
        break
    frame = imutils.resize(frame, 1200)
    Background = imutils.resize(Background, 1200)
    # subtract background
    
    subtract = cv.absdiff(frame, Background)
    img_gray = cv.cvtColor(subtract, cv.COLOR_BGR2GRAY)
    _, subtract = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY)
    print(subtract.shape)
    cv.imshow('subtract', subtract)
    cv.waitKey(0)
    
