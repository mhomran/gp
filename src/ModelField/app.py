# importing the module
import cv2 as cv
from model_field import *

clicks = []
img = None

# driver function
if __name__=="__main__":

  # reading the image
  img = cv.imread('../../data/imgs/stitched/sample1.png')

  ModelField(img)

  cv.destroyAllWindows()


