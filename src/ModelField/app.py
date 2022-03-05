# importing the module
import cv2 as cv
from model_field import *

clicks = []
img = None

# driver function
if __name__=="__main__":

  # reading the image
  cap = cv.VideoCapture("../../data/videos/cuda_output.avi")
  ret, frame = cap.read()
  
  ModelField(frame)

  cv.destroyAllWindows()


