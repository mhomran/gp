from matplotlib.pyplot import draw
from PlayerDetection.PlayerDetection import PlayerDetction
from PlayerDetection.ImageClass import ImageClass
from ModelField.model_field import ModelField
import numpy as np
import pickle
import cv2 as cv
import copy

IMG = ImageClass('../../output.avi')
PD = PlayerDetction()

# ret, frame, frameId = IMG.readFrame()
# MF = ModelField(frame)
# particles = MF._get_particles()

particles_ORG = {}
with open('file.pkl', 'rb') as f:
    particles_ORG = pickle.load(f)

particles = copy.deepcopy(particles_ORG)

# roi = image[startY:endY, startX:endX]


def _get_roi(B):
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


def getRatio(frame):
    number_of_white_pix = np.sum(frame == 255)
    number_of_black_pix = np.sum(frame == 0)
    percentage = round(number_of_white_pix /
                       (number_of_white_pix+number_of_black_pix), 2)

    return percentage


def getDecision(p1, p2, p3):
    if(p1 < .15):
        return False, 0
    if(p2 < .15):
        return False, 0
    if(p3 < .15):
        return False, 0

    return True, round((p1+p2+p3)/3, 2)


def IOU(B1, B2):

    print("-----")
    print(B1.br)
    print(B1.tl)
    dx = min(B1.br[0], B2.br[0]) - max(B1.tl[0], B2.tl[0])
    dy = min(B1.br[1], B2.br[1]) - max(B1.tl[1], B2.tl[1])

    area1 = (B1.br[0]-B1.tl[0])*(B1.br[1]-B1.tl[1])
    area2 = (B2.br[0]-B2.tl[0])*(B2.br[1]-B2.tl[1])

    print('--area--')
    print(area1)
    print(area2)
    intersection = dx*dy
    union = area1+area2-intersection

    assert union != 0
    print("-------")
    print(intersection)
    print(union)
    print(round((intersection/union), 2))

    return round((intersection/union), 2)


def applyNonMax(particles):

    i = 0
    while(i < len(particles)):
        j = i+1
        while(j < len(particles)):
            iou = IOU(particles[i].B, particles[j].B)
            if(iou > .5):
                particles.pop(j)
                print('---remove---')
                j = j-1
            j = j+1
        i = i+1
    print(f"particles: {len(particles)}")
    return particles


while True:

    ret, frame, frameId = IMG.readFrame()

    if frame is None:
        break

    fgMask = PD.subBG(frame)
    IMG.writeTxt(frame)
    IMG.showImage(frame, "Frame")

    if(frameId > 300):
        openingImg = PD.opening(fgMask)
        PD.getContours(openingImg)
        contourFrame = frame.copy()
        MFfrmae = frame.copy()
        PD.drawContours(contourFrame)

        IMG.showImage(contourFrame, "contours")
        BB = PD.getBoundingBoxes()

        # loop on BBs
        for index, B in enumerate(BB):
            MFBB = {}
            Sx = B.tl[0]
            Ex = B.br[0]

            Sy = B.tl[1]
            Ey = B.br[1]

            # get all particle in BB
            for y in range(Sy, Ey+1):
                for x in range(Sx, Ex+1):
                    if particles.get((x, y)):
                        MFBB[(x, y)] = particles[(x, y)]

            # draw BB
            cv.rectangle(frame, B.tl,
                         B.br, (0, 255, 0), 1)
            IMG.showImage(frame, "BB")

            # get candidate particle
            NonMax = []
            for particle in list(MFBB):
                roi1, roi2, roi3 = _get_roi(MFBB[particle].B)
                pRo1 = getRatio(fgMask[roi1[0]:roi1[1], roi1[2]:roi1[3]])
                pRo2 = getRatio(fgMask[roi2[0]:roi2[1], roi2[2]:roi2[3]])
                pRo3 = getRatio(fgMask[roi3[0]:roi3[1], roi3[2]:roi3[3]])

                decision, ratio = getDecision(pRo1, pRo2, pRo3)

                if(decision):
                    MFBB[particle].ratio = ratio
                    NonMax.append(MFBB[particle])

            # apply nonmax supression
            NonMax.sort(key=lambda x: x.ratio, reverse=True)
            NonMax = applyNonMax(NonMax)

            # draw particles
            for particle in NonMax:
                cv.rectangle(MFfrmae, particle.B.tl,
                             particle.B.br, (255, 0, 0), 1)

            IMG.showImage(MFfrmae, "MFBB")
            keyboard = cv.waitKey(0)
            if keyboard == 'q' or keyboard == 27:
                break

    wait = 1
    if(frameId > 300):
        wait = 0
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break

# save model
# with open('file.pkl', 'wb') as f:
#     pickle.dump(particles, f)
# f.close()


#  for index, B in enumerate(BB):
#             start_X = B.tl[0]
#             start_Y = B.tl[1]
#             end_X = B.br[0]
#             end_Y = B.br[1]
#             IMG.showImage(
#                 fgMask[start_Y:end_Y, start_X:end_X], str(index), (end_X-start_X)*4)

# for x in list(particles)[0:3]:
#     roi1, roi2, roi3 = _get_roi(particles[x].B)
#     # print(particles[x].q)      # (grid)
#     # print(particles[x].q_img)  # (frame)
#     print(particles[x].B.tl)
#     print(particles[x].B.br)
#     print(roi1)
#     print(roi2)
#     print(roi3)
