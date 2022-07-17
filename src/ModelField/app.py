import sys
sys.path.append(sys.path[0]+"/../")

print(sys.path)

import cv2 as cv
from Canvas.canvas import Canvas
from model_field import ModelField
import imutils

clicks = []
img = None
done = False
frame = None
gui_img_shape = None
top_view = None
mf = None

def gui2orig(p):
    x = p[0] * frame.shape[1] // gui_img_shape[1]
    y = p[1] * frame.shape[0] // gui_img_shape[0]
    return (x, y)

def click_event(event, x, y, flags=None, params=None):
  x, y = gui2orig((x, y))
  particle = mf.get_nearest_particle((x, y))
  if particle:
    cv.circle(frame, particle.q_img, 3, (0,0,255), 3)
    cv.circle(top_view, particle.q, 3, (0,0,255), 3)
  
# driver function
if __name__=="__main__":
  top_view = cv.imread("h.png")
  
  cap = cv.VideoCapture("two_mins.avi")
  _, frame = cap.read()

  frame_temp = imutils.resize(frame, 1700)
  gui_img_shape = frame_temp.shape

  canvas = Canvas(frame_temp.shape, top_view_shape=top_view.shape)
  mf = ModelField(frame, samples_per_meter=1, canvas=canvas)
  canvas.set_callback(click_event)

  while True:
      top_temp = imutils.resize(top_view, 500)
      frame_temp = imutils.resize(frame, 1700)
      canvas.show_canvas(frame_temp, top_temp)
      
      if cv.waitKey(1) == 27:
        break


  cap.release()
  cv.destroyAllWindows()
