from canvas import Canvas
import cv2 as cv
import imutils

def main():
  frame_fn = "/home/mhomran/gp/src/two_mins.avi"

  frame_cap = cv.VideoCapture(frame_fn)

  top_view = cv.imread("h.png")

  canvas = Canvas()
  while True:
    _, frame = frame_cap.read()
    frame = imutils.resize(frame, width=1700)
    
    canvas.show_canvas(frame, top_view=top_view, 
    status="Trackista", status_color=(255, 255, 255))
    if (cv.waitKey(1) == 27):
      break
  
  frame_cap.release()

main()