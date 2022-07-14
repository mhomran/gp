import cv2 as cv
from model_field import ModelField

clicks = []
img = None
done = False
frame = None
top_view = None
mf = None

def click_event(event, x, y, flags=None, params=None):
  print(x, y)
  particle = mf.get_nearest_particle((x, y))
  if particle:
    cv.circle(frame, particle.q_img, 3, (0,0,255), 3)
    cv.circle(top_view, particle.q, 3, (0,0,255), 3)
  
# driver function
if __name__=="__main__":
  top_view = cv.imread("h.png")
  
  cap = cv.VideoCapture("two_mins.avi")
  _, frame = cap.read()
  
  mf = ModelField(frame, samples_per_meter=1)

  cv.namedWindow("frame", cv.WINDOW_NORMAL)
  cv.namedWindow("top_view")
  cv.resizeWindow("frame", 1200, 500)
  
  cv.setMouseCallback('frame', click_event)
  while True:
      cv.imshow('frame', frame)
      cv.imshow('top_view', top_view)
      
      if cv.waitKey(1) == 27:
        break


  cap.release()
  cv.destroyAllWindows()
