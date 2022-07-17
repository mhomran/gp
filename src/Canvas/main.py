from canvas import Canvas
import cv2 as cv
import imutils

def main():
  frame_fn = "/home/mhomran/gp/src/two_mins.avi"

  frame_cap = cv.VideoCapture(frame_fn)

  top_view = cv.imread("h.png")
  top_view = imutils.resize(top_view, 500)

  _, frame = frame_cap.read()
  frame = imutils.resize(frame, width=1700)

  canvas = Canvas(frame.shape, top_view.shape)
  while True:
    _, frame = frame_cap.read()
    frame = imutils.resize(frame, width=1700)
    
    canvas.show_canvas(frame, top_view=top_view, status="Trackista")
    if (cv.waitKey(1) == 27):
      break
  
  frame_cap.release()

main()