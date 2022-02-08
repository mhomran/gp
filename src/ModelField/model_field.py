import cv2 as cv
import numpy as np
from enum import Enum
import imutils

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
T_GOAL_CM = 244 # goal real height in cm
T_PLAYER_CM = 190 # Player real height in cm
SOCCER_HEIGHT_M = 68 # Soccer pitch in meters
SOCCER_WIDTH_M = 108

# TODO: configure
THICKNESS = 3 # thickness of drawings
SAMPLES_PER_METER = 1
PIXELS_PER_METER = 10
PLAYER_ASPECT_RATIO = 9 / 16
GUI_WIDTH = 1500

class BoundingBox:
  def __init__(self, tl, br):
    self.tl = tl
    self.br = br

  def draw(self, img):
    cv.rectangle(img, self.tl, self.br, RED_COLOR, THICKNESS)    

class Particle:
  def __init__(self, q, q_img, B, a=None, e=None):
    self.q = q # particle position
    self.q_img = q_img # particle position on the image
    self.B = B # bounding box
    self.a = a # appearance model
    self.e = e # the probability of containing a player

class GuiState(Enum):
  STATE_CORNERS = 1,
  STATE_GOAL = 2,

class ModelField:
  def __init__(self, img):
    self.original_img = img.copy()
    self.original_img_without_BBs = img.copy()

    self.gui_img = img.copy()
    self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)
    
    self.grid = cv.imread("../../data/imgs/pitch/h.png") # modelfield image (Top view)
    self.grid_res_w = self.grid.shape[1]
    self.px_per_m_w = self.grid_res_w // SOCCER_WIDTH_M
    self.sample_inc_w = self.px_per_m_w // SAMPLES_PER_METER
    self.grid_res_h = self.grid.shape[0]
    self.px_per_m_h = self.grid_res_h // SOCCER_HEIGHT_M
    self.sample_inc_h = self.px_per_m_h // SAMPLES_PER_METER

    self.clicks = []
    self.s = None # particles
    self.L_horizon = None # the horizon line (m, c)
    self.hcam = None # camera height
    self.H = None # homography matrix
    self.gui_state = GuiState.STATE_CORNERS

    self._write_hint("choose the bottom left corner")

    # cv.namedWindow("GUI", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.namedWindow("GUI")
    cv.setMouseCallback('GUI', self.click_event)
    while True:
      cv.imshow('GUI', self.gui_img)
      if cv.waitKey(100) == ord('q'):
        break
  
  def _write_hint(self, msg, color=(0,0,0)):    
    cv.rectangle(self.gui_img, (10, 2), (300,20), (255,255,255), -1)
    cv.putText(self.gui_img, msg, (15, 15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5 , color)
  
  def _gui2orig(self, p):
    x = p[0] * self.original_img.shape[1] // self.gui_img.shape[1]
    y = p[1] * self.original_img.shape[0] // self.gui_img.shape[0] 
    return (x, y)

  def click_event(self, event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
      self.clicks.append(self._gui2orig((x, y)))
            
      if self.gui_state == GuiState.STATE_CORNERS:
        if len(self.clicks) <= 4:
          curr_click = self.clicks[-1]
          self.original_img = cv.circle(self.original_img, curr_click, THICKNESS, RED_COLOR, cv.FILLED)
          if len(self.clicks) == 1:
            self._write_hint("choose the upper left corner")
          elif len(self.clicks) == 2:
            self._write_hint("choose the upper right corner")
          elif len(self.clicks) == 3:
            self._write_hint("choose the bottom right corner")

        if len(self.clicks) == 4:
          pts1 = np.float32(self.clicks)
          pts2 = np.float32([[0, 0], [self.grid_res_w, 0], 
          [self.grid_res_w, self.grid_res_h], [0, self.grid_res_h]])
          self.H = cv.getPerspectiveTransform(pts2, pts1)

          A, B = (self.clicks[0], self.clicks[3])
          C, D = (self.clicks[1], self.clicks[2])
          self.L_horizon = self._calculate_horizon(A, B, C, D)
            
          self._write_hint("choose the bottom of the post")

          self.clicks = []
          self.gui_state = GuiState.STATE_GOAL

      elif self.gui_state == GuiState.STATE_GOAL:
        if len(self.clicks) == 1:
          self._write_hint("choose the top of the post")

        elif len(self.clicks) == 2:
          u_bottom = self.clicks[0]
          u_top = self.clicks[1]
          cv.line(self.original_img, u_bottom, u_top, RED_COLOR, THICKNESS)

          # equation (3.3)
          dst_u_bt = self._euclidean_distance(u_bottom, u_top)
          dst_u_L = self._min_dist_line_point(self.L_horizon, u_bottom)
          self.hcam = (dst_u_L * T_GOAL_CM) / dst_u_bt

          self.s = self._construct_modelfield_img(self.H)

          cv.imwrite("modelfield.png", self.grid)
          cv.imwrite("result.png", self.original_img)
          self._write_hint("Done", RED_COLOR)



  def _min_dist_line_point(self, L, p):
    # Description: get the minimum distance between a point and a line.
    # Input:
    #   L: list contains the slope and the constant of the line 
    #   p: the point 
    m = L[0]
    c = L[1]
    if m == 0:
      return np.abs(c - p[1]) 
    else:
      # TODO: implement
      perp_m = -(1/m)

  def _euclidean_distance(self, p1, p2):
    dst = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    return dst 

  def _calculate_horizon(self, A, B, C, D):
    # Description: get the intersection between the two lines
    # the first line passes through A, B. Its slope a, constant b.
    # the second line passes through C, D. Its slope c, constant d.
    # then draw a horizontal line.

    # y = ax + b
    a = (A[1] - B[1])/(A[0] - B[0])
    b = A[1] - a * A[0]

    # y = cx + d
    c = (C[1] - D[1])/(C[0] - D[0])
    d = C[1] - c * C[0]

    # TODO: check if there's an intersection
    # intersection (solve for x)
    x = int((d - b) / (a - c))
    y = int(a * x + b)

    cv.line(self.original_img, (0, y), (self.original_img.shape[1], y), RED_COLOR, THICKNESS)

    return 0, y

  def _construct_modelfield_img(self, H):
    s_total = []

    for row in range(self.sample_inc_h, self.grid_res_h, self.sample_inc_h):
      for col in range(self.sample_inc_w, self.grid_res_w, self.sample_inc_w):
        q = (col, row)

        q_img = cv.perspectiveTransform(np.array([[q]], np.float32), H)
        q_img = tuple(q_img.squeeze())
        q_img = (int(q_img[0]), int(q_img[1]))

        # equation (3.3)
        BB_height = (self._min_dist_line_point(self.L_horizon, q_img) * T_PLAYER_CM) / self.hcam
        BB_width = BB_height * PLAYER_ASPECT_RATIO
        tl = (int(q_img[0] - BB_width // 2), int(q_img[1] - BB_height)) 
        br = (int(q_img[0] + BB_width // 2), int(q_img[1])) 
        
        B = BoundingBox(tl, br)
        a = self.original_img_without_BBs[tl[1]:int(tl[1]+BB_height), tl[0]:int(tl[0]+BB_width)]
        # save the bounding boxes. Warning: computational intensive 
        # cv.imwrite(f"BBs/BB_{row}_{col}.png", a) 
        s = Particle(q, q_img, B, a)

        s_total.append(s)

        B.draw(self.original_img)
        self.grid = cv.circle(self.grid, q, 2, BLUE_COLOR, cv.FILLED)
        self.original_img = cv.circle(self.original_img, q_img, THICKNESS, BLUE_COLOR, cv.FILLED)

    return s_total
