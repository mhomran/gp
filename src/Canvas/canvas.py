import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Canvas:
  def __init__(self, frame_pos=(50, 30), 
  top_view_pos=(600, 470), status_pos=(50, 500)) -> None:

    self.template = cv.imread("Canvas/template.png")
    self.canvas = None

    self.frame_pos = frame_pos
    self.top_view_pos = top_view_pos
    self.status_pos = status_pos

    self.font = ImageFont.truetype("Canvas/font.ttf", 50)



  def _clean(self):
    self.canvas = self.template.copy()
    self.canvas = Image.fromarray(self.canvas)

  def show_canvas(self, frame, top_view=None, 
  status=None, status_color=(0, 0, 0)):
    self._clean()

    frame = Image.fromarray(frame)

    self.canvas.paste(frame, self.frame_pos)
    if top_view is not None:
      top_view = Image.fromarray(top_view)
      self.canvas.paste(top_view, self.top_view_pos)
    if status is not None:
      canvas_draw = ImageDraw.Draw(self.canvas)
      canvas_draw.text(self.status_pos, status, status_color, self.font)

    img = np.asarray(self.canvas)
    cv.imshow("Trackista", img)
