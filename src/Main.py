from matplotlib.pyplot import draw
from PlayerDetection.PlayerDetection import PlayerDetction
from PlayerDetection.ImageClass import ImageClass
from ModelField.model_field import ModelField
import pickle
import cv2 as cv


IMG = ImageClass('../../output.avi')
PD = PlayerDetction()


particles = {}
with open('file.pkl', 'rb') as f:
    particles = pickle.load(f)

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


# for x in list(particles)[0:3]:
#     roi1, roi2, roi3 = _get_roi(particles[x].B)
#     # print(particles[x].q)      # (grid)
#     # print(particles[x].q_img)  # (frame)
#     print(particles[x].B.tl)
#     print(particles[x].B.br)
#     print(roi1)
#     print(roi2)
#     print(roi3)


while True:

    ret, frame, frameId = IMG.readFrame()

    if frame is None:
        break

    fgMask = PD.subBG(frame)
    IMG.writeTxt(frame)
    IMG.showImage(frame, "Frame")

    if(frameId > 300):
        MFBB = []
        openingImg = PD.opening(fgMask)
        PD.getContours(openingImg)
        PD.drawContours(frame)
        IMG.showImage(frame, "contours")
        BB = PD.getBoundingBoxes()
        for index, B in enumerate(BB):
            Sx = B.tl[0]
            Ex = B.br[0]

            Sy = B.tl[1]
            Ey = B.br[1]

            for y in range(Sy, Ey+1):
                for x in range(Sx, Ex+1):
                    if particles.get((x, y)):
                       # print(f'({x},{y}) is found')
                        MFBB.append(particles[(x, y)])
                    # else:
                    #     print(f'({x},{y}) not found')

        print('----')
        print(len(MFBB))
            # keyboard = cv.waitKey(0)
            # if keyboard == 'q' or keyboard == 27:
            #     break

    wait = 1
    if(frameId > 300):
        wait = 0
    keyboard = cv.waitKey(1)
    if keyboard == 'q' or keyboard == 27:
        break


# img = cv.imread('../data/imgs/stitched/sample1.png')
# MF = ModelField(img)
# particles = MF._get_particles()
# f = open("file.pkl","wb")
# pickle.dump(particles,f)
# f.close()


#  for index, B in enumerate(BB):
#             start_X = B.tl[0]
#             start_Y = B.tl[1]
#             end_X = B.br[0]
#             end_Y = B.br[1]
#             IMG.showImage(
#                 fgMask[start_Y:end_Y, start_X:end_X], str(index), (end_X-start_X)*4)
