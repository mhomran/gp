# importing the module
import cv2 as cv
from model_field import *

clicks = []
img = None

# driver function
if __name__=="__main__":

  # reading the image
  cap = cv.VideoCapture("output.avi")
  ret, frame = cap.read()
  
  mf_gui_clicks = [ (311, 110), (616, 101), (922, 103), # the three top corners
                  (1196, 223), (619, 265), (27, 240), # the three bottom corners
                  (195, 162), (193, 142), # the left post corners
                  (1035, 152), (1037, 132) ] # the right post corners

  mf = ModelField(frame, 1, clicks=mf_gui_clicks)
  particle = mf.get_nearest_particle((1200, 500))

  cap.release()
<<<<<<< HEAD
  cv.destroyAllWindows()
=======
  cv.destroyAllWindows()
>>>>>>> cb416d847fc38312f1670960f1d171e9cd562989
