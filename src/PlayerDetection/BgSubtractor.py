import cv2 as cv
import imutils

cap = cv.VideoCapture('../output.avi')

BACK_SUB_HISTORY = 500
BACK_SUB_THRE = 16
BACK_SUB_DETECT_SHADOW = True

backSub = cv.createBackgroundSubtractorMOG2(
    BACK_SUB_HISTORY, BACK_SUB_THRE, BACK_SUB_DETECT_SHADOW)

limit = 990


while True:
    ret, frame = cap.read()
    frameId = int(cap.get(1))
    if frame is None:
        break

    # OrginalImage
    frameDis = imutils.resize(frame, 1200)

    cv.rectangle(frameDis, (10, 2), (100, 20), (255, 255, 255), -1)
    cv.putText(frameDis, str(cap.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    cv.imshow('OrgFrame', frameDis)

    # BackgroundSubtraction
    mask = backSub.apply(frame)
    mask = imutils.resize(mask, 1200)
    cv.imshow('mask', mask)

    # Background
    bgIMG = backSub.getBackgroundImage()
    bgIMGDispaly = imutils.resize(bgIMG, 1200)
    cv.imshow('background', bgIMGDispaly)

    if(frameId == limit):
        cv.imwrite('back_ground.png', bgIMG)
        exit()

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
