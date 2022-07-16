import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class Canvas:
  def __init__(self, frame_shape,top_view_shape, frame_pos=(50, 30),
  top_view_pos=(680, 555), status_pos=(60, 510),info_pos=(65,620)) -> None:

    self.template = cv.imread("Canvas/template.png")
    self.canvas = None

    self.frame_shape = frame_shape
    self.frame_pos = frame_pos
    self.top_view_shape = top_view_shape
    self.top_view_pos = top_view_pos
    self.status_pos = status_pos
    self.status_bg_bb = [50, 500, 650, 560]
    self.status_bg_rad = 5
    self.info_pos = info_pos
    self.info_bg_bb = [50, 600, 600, 860]
    self.info_bg_rad = 5
    self.font = ImageFont.truetype("Canvas/font.ttf", 40)
    self.infofont = ImageFont.truetype("Canvas/font.ttf", 20)
    self.callback = None
    self.top_view_callback = None
    cv.namedWindow("Trackista")
    cv.setMouseCallback("Trackista", self.click_event)


  def _clean(self):
    self.canvas = self.template.copy()
    self.canvas = Image.fromarray(self.canvas)

  def show_canvas(self, frame, top_view=None, 
  status=None, status_color=(0, 0, 0),info = None, info_color = (255,255,255)):
    self._clean()

    frame = Image.fromarray(frame)

    self.canvas.paste(frame, self.frame_pos)
    if top_view is not None:
      top_view = Image.fromarray(top_view)
      self.canvas.paste(top_view, self.top_view_pos)
    if status is not None:
      canvas_draw = ImageDraw.Draw(self.canvas)
      canvas_draw.rounded_rectangle(self.status_bg_bb, self.status_bg_rad,
      (255, 255, 255))
      canvas_draw.text(self.status_pos, status, status_color, self.font)
    if info is not None:
      canvas_draw = ImageDraw.Draw(self.canvas)
      canvas_draw.rounded_rectangle(self.info_bg_bb, self.info_bg_rad,
      (0, 0, 0))
      canvas_draw.text(self.info_pos, info, info_color, self.infofont)
      
    img = np.asarray(self.canvas)
    cv.imshow("Trackista", img)    

  def set_callback(self, callback):
    self.callback = callback
    cv.setMouseCallback("Trackista", self.click_event)    
  def click_event(self, event, x, y, flags=None, params=None):
    if self.callback is not None:
      h, w, _ = self.frame_shape
      x_os, y_os = self.frame_pos
      if x_os < x < (x_os+w) and y_os < y < (y_os+h):
        x = x - x_os
        y = y - y_os
        self.callback(event, x, y, flags, params)

  def set_top_view_callback(self,callback):
    self.top_view_callback = callback
    cv.setMouseCallback("Trackista", self.top_view_click_event)
  def top_view_click_event(self, event, x, y, flags=None, params=None):
    if self.top_view_callback is not None:
      h, w, _ = self.top_view_shape
      x_os, y_os = self.top_view_pos
      if x_os < x < (x_os+w) and y_os < y < (y_os+h):
        x = x - x_os
        y = y - y_os
        self.top_view_callback(event, x, y, flags, params)