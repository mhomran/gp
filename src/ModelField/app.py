# importing the module
import cv2 as cv
import imutils
from model_field import *

clicks = []
img = None

# driver function
if __name__=="__main__":

  # reading the image
  img = cv.imread('sample2.png')
  cv.imwrite('sample2_resized.png', img)

  ModelField(img)

  # close the window
  cv.destroyAllWindows()


