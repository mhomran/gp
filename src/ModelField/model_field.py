import cv2 as cv
import numpy as np
from enum import Enum
import imutils

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
T_GOAL_CM = 244
T_PLAYER_CM = 190

# TODO: configure
THICKNESS = 15
SOCCER_HEIGHT_M = 68
SOCCER_WIDTH_M = 108
SAMPLES_PER_METER = 1
PIXELS_PER_METER = 10
GOAL_POST_H = 40
GOAL_POST_W = 30

SOCCER_RES_W = SOCCER_WIDTH_M * PIXELS_PER_METER
SOCCER_RES_H = SOCCER_HEIGHT_M * PIXELS_PER_METER
PLAYER_ASPECT_RATIO = 9 / 16

class BoundingBox:
  def __init__(self, tl, br):
    self.tl = tl
    self.br = br

  def draw(self, img):
    cv.rectangle(img, self.tl, self.br, RED_COLOR, THICKNESS)
    

class Particle:
  def __init__(self, q, q_img, B):
    self.q = q # particle position
    self.q_img = q_img # particle position on the image
    self.B = B # bounding box
    self.a = None # appearance model
    self.e = None # the probability of containing a player

class GuiState(Enum):
  STATE_CORNERS = 1,
  STATE_GOAL = 2,

class ModelField:
  def __init__(self, img):
    self.original_img = img.copy()

    # self.gui_img = imutils.resize(img, width=1280)
    self.gui_img = img.copy()
    self.gui_img = cv.copyMakeBorder(self.gui_img, 50, 0, 0, 0, cv.BORDER_CONSTANT, value=0)
    self.clicks = []
    self.mf = None # modelfield image (Top view)
    self.s = None # particles
    self.L_horizon = None # the horizon line (m, c)
    self.hcam = None # camera height
    self.H = None # homography matrix
    self.gui_state = GuiState.STATE_CORNERS



    cv.namedWindow("GUI", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.setMouseCallback('GUI', self.click_event)
    while True:
      cv.imshow('GUI', self.gui_img)
      if cv.waitKey(100) == ord('q'):
        break

  def click_event(self, event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
      self.clicks.append((x, y))
            
      if self.gui_state == GuiState.STATE_CORNERS:
        if len(self.clicks) <= 4:
          curr_click = self.clicks[-1]
          self.gui_img = cv.circle(self.gui_img, curr_click, THICKNESS, RED_COLOR, cv.FILLED)
        
        if len(self.clicks) == 4:
          pts1 = np.float32(self.clicks)
          pts2 = np.float32([[0, 0], [SOCCER_RES_W, 0], 
          [SOCCER_RES_W, SOCCER_RES_H], [0, SOCCER_RES_H]])
          self.H = cv.getPerspectiveTransform(pts2, pts1)

          A, B = (self.clicks[0], self.clicks[3])
          C, D = (self.clicks[1], self.clicks[2])
          self.L_horizon = self._calculate_horizon(A, B, C, D)

          self.clicks = []
          self.gui_state = GuiState.STATE_GOAL


      elif self.gui_state == GuiState.STATE_GOAL:
        if len(self.clicks) == 2:
          u_bottom = self.clicks[0]
          u_top = self.clicks[1]
          cv.line(self.gui_img, u_bottom, u_top, RED_COLOR, THICKNESS)

          dst_u_bt = self._euclidean_distance(u_bottom, u_top)
          dst_u_L = self._min_dist_line_point(self.L_horizon, u_bottom)
          self.hcam = (dst_u_L * T_GOAL_CM) / dst_u_bt

          self.s = self._construct_modelfield_img(self.H)

          cv.imwrite("modelfield.png", self.mf)
          cv.imwrite("result.png", self.gui_img)


  def _min_dist_line_point(self, L, p):
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
    # y = ax + b
    a = (A[1] - B[1])/(A[0] - B[0])
    b = A[1] - a * A[0]

    # y = cx + d
    c = (C[1] - D[1])/(C[0] - D[0])
    d = C[1] - c * C[0]

    # TODO: check if there's an intersection
    # intersection 
    x = int((d - b) / (a - c))
    y = int(a * x + b)

    cv.line(self.gui_img, (0, y), (self.gui_img.shape[1], y), RED_COLOR, THICKNESS)

    return 0, y

  def _construct_modelfield_img(self, H):
    s_total = []

    sample_inc = int(PIXELS_PER_METER / SAMPLES_PER_METER)
    self.mf = np.zeros((SOCCER_RES_H, SOCCER_RES_W, 3), np.uint8)

    for row in range(sample_inc, SOCCER_RES_H, sample_inc):
      for col in range(sample_inc, SOCCER_RES_W, sample_inc):
        q = (col, row)

        q_img = cv.perspectiveTransform(np.array([[q]], np.float32), H)
        q_img = tuple(q_img.squeeze())
        q_img = (int(q_img[0]), int(q_img[1]))

        BB_height = (self._min_dist_line_point(self.L_horizon, q_img) * T_PLAYER_CM) / self.hcam
        BB_width = BB_height * PLAYER_ASPECT_RATIO
        tl = (int(q_img[0] - BB_width // 2), int(q_img[1] - BB_height)) 
        br = (int(q_img[0] + BB_width // 2), int(q_img[1])) 
        
        B = BoundingBox(tl, br)
        a = self.original_img[tl[1]:int(tl[1]+BB_height), tl[0]:int(tl[0]+BB_width)]
        # a = cv.resize(a, (int(T_PLAYER_CM*PLAYER_ASPECT_RATIO), T_PLAYER_CM))
        cv.imwrite(f"BBs/BB_{row}_{col}.png", a)
        s = Particle(q, q_img, B)
        

        s_total.append(s)


        B.draw(self.gui_img)
        self.mf = cv.circle(self.mf, q, 2, BLUE_COLOR, cv.FILLED)
        self.gui_img = cv.circle(self.gui_img, q_img, THICKNESS, BLUE_COLOR, cv.FILLED)

    return s_total

