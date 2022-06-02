import cv2 as cv
import numpy as np
from enum import Enum
import imutils
import pickle
from scipy.optimize import fsolve

RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
T_GOAL_CM = 244  # goal real height in cm
T_PLAYER_CM = 190  # Player real height in cm
SOCCER_HEIGHT_M = 68  # Soccer pitch in meters
SOCCER_WIDTH_M = 108

# TODO: configure
THICKNESS = 3  # thickness of drawings
PIXELS_PER_METER = 10
PLAYER_ASPECT_RATIO = 9 / 16
GUI_WIDTH = 1200


class BoundingBox:
    def __init__(self, tl, br):
        self.tl = tl
        self.br = br

    def draw(self, img):
        cv.rectangle(img, self.tl, self.br, RED_COLOR, THICKNESS)


class Particle:
    def __init__(self, q, q_img, B, a=None, e=None):
        self.q = q  # particle position (grid)
        self.q_img = q_img  # particle position on the image (frame)
        self.B = B  # bounding box
        self.a = a  # appearance model
        self.e = e  # the probability of containing a player


class GuiState(Enum):
    STATE_CORNERS = 1,
    STATE_GOAL = 2,


class ModelField:
    def __init__(self, img, samples_per_meter):
        self.original_img = img.copy()
        self.original_img_without_BBs = img.copy()

        self.gui_img = img.copy()
        self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)

        # modelfield image (Top view)
        self.grid = cv.imread("../data/imgs/pitch/h.png")
        self.grid_res_w = self.grid.shape[1]
        self.px_per_m_w = self.grid_res_w // SOCCER_WIDTH_M
        self.sample_inc_w = int(self.px_per_m_w // samples_per_meter)
        self.grid_res_h = self.grid.shape[0]
        self.px_per_m_h = self.grid_res_h // SOCCER_HEIGHT_M
        self.sample_inc_h = int(self.px_per_m_h // samples_per_meter)

        self.clicks = []
        self.s = None  # particles
        self.L_horizon = None  # the horizon line (m, c)
        self.hcam = None  # camera height
        self.H = None  # homography matrix
        self.gui_state = GuiState.STATE_CORNERS
        self.done = False

        self._write_hint("choose the upper left corner")

        # cv.namedWindow("GUI", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.namedWindow("GUI")
        cv.setMouseCallback('GUI', self.click_event)
        while True:
            cv.imshow('GUI', self.gui_img)
            if self.done: 
              cv.waitKey(2000)
              cv.destroyAllWindows()
              break
            cv.waitKey(1)

    def _write_hint(self, msg, color=(0, 0, 0)):
        cv.rectangle(self.gui_img, (10, 2), (300, 20), (255, 255, 255), -1)
        cv.putText(self.gui_img, msg, (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

    def _gui2orig(self, p):
        x = p[0] * self.original_img.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.original_img.shape[0] // self.gui_img.shape[0]
        return (x, y)

    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            self.clicks.append(self._gui2orig((x, y)))

            if self.gui_state == GuiState.STATE_CORNERS:
                if len(self.clicks) < 6:
                    curr_click = self.clicks[-1]
                    self.original_img = cv.circle(
                        self.original_img, curr_click, THICKNESS, RED_COLOR, cv.FILLED)
                    if len(self.clicks) == 1:
                        self._write_hint("choose the upper center corner")
                    elif len(self.clicks) == 2:
                        self._write_hint("choose the upper right corner")
                    elif len(self.clicks) == 3:
                        self._write_hint("choose the bottom right corner")
                    elif len(self.clicks) == 4:
                        self._write_hint("choose the bottom center corner")
                    elif len(self.clicks) == 5:
                        self._write_hint("choose the bottom left corner")

                elif len(self.clicks) == 6:
                    half_w = self.grid_res_w // 2
                    pts1 = np.float32(self.clicks[:2] + self.clicks[4:])
                    pts2 = np.float32([[0, 0], [half_w, 0],
                                        [half_w, self.grid_res_h], 
                                        [0, self.grid_res_h]])
                    self.lH = cv.getPerspectiveTransform(pts2, pts1)
                    
                    A, B, C, D = pts1
                    self.lL_horizon = self._calculate_horizon(A, B, C, D)
                    
                    pts1 = np.float32(self.clicks[1:3] + self.clicks[3:5])
                    pts2 = np.float32([[half_w, 0], [self.grid_res_w, 0],
                                        [self.grid_res_w, self.grid_res_h], 
                                        [half_w, self.grid_res_h]])
                    self.rH = cv.getPerspectiveTransform(pts2, pts1)

                    A, B, C, D = pts1
                    self.rL_horizon = self._calculate_horizon(A, B, C, D)
                    
                    self._write_hint("choose the bottom of the left post")

                    self.clicks = []
                    self.gui_state = GuiState.STATE_GOAL

            elif self.gui_state == GuiState.STATE_GOAL:
                if len(self.clicks) < 4:
                    if len(self.clicks) == 1:
                        self._write_hint("choose the top of the left post")
                    if len(self.clicks) == 2:
                        self._write_hint("choose the bottom of the right post")
                    if len(self.clicks) == 3:
                        self._write_hint("choose the top of the right post")

                elif len(self.clicks) == 4:
                    u_bottom, u_top = self.clicks[0:2]
                    cv.line(self.original_img, u_bottom,
                            u_top, RED_COLOR, THICKNESS)

                    # equation (3.3)
                    dst_u_bt = self._euclidean_distance(u_bottom, u_top)
                    dst_u_L = self._min_dist_line_point(self.lL_horizon, u_bottom)
                    self.lhcam = (dst_u_L * T_GOAL_CM) / dst_u_bt
                    
                    u_bottom, u_top = self.clicks[2:]
                    cv.line(self.original_img, u_bottom,
                            u_top, RED_COLOR, THICKNESS)

                    # equation (3.3)
                    dst_u_bt = self._euclidean_distance(u_bottom, u_top)
                    dst_u_L = self._min_dist_line_point(self.rL_horizon, u_bottom)
                    self.rhcam = (dst_u_L * T_GOAL_CM) / dst_u_bt

                    self.s = self._construct_modelfield_img()

                    cv.imwrite("modelfield.png", self.grid)
                    cv.imwrite("result.png", self.original_img)
                    self._write_hint("Done", RED_COLOR)
                    self.done = True

    def _min_dist_line_point(self, L, p):
        # Description: get the minimum distance between a point and a line.
        # Input:
        #   L: list contains the slope and the constant of the line
        #   p: the point
        m = L[0]
        c = L[1]
        x0 = p[0]
        y0 = p[1]
        d = np.abs((m*x0-y0+c))/np.sqrt(m**2 + 1)
        return d

    def _euclidean_distance(self, p1, p2):
        dst = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return dst

    def _get_intersect(self, A, B, C, D):
        # Description: get the intersection between the two lines
        # the first line passes through A, B. Its slope a, constant b.
        # the second line passes through C, D. Its slope c, constant d.

        # y = ax + b
        a = (A[1] - B[1])/(A[0] - B[0])
        b = A[1] - a * A[0]

        # y = cx + d
        c = (C[1] - D[1])/(C[0] - D[0])
        d = C[1] - c * C[0]

        # same slope (no intersection) ?
        if a == c: return False, None

        # intersection (solve for x)
        x = (d - b) / (a - c)
        y = (a * x + b)

        return True, (x, y)

    def _calculate_horizon(self, A, B, C, D):
        # Description: given the top left A, top right B,
        # bottom right C, bottom left D, we need to calculate 
        # the horizon of the perspective.
        # Input:
        #   A: top left
        #   B: top right
        #   C: bottom right
        #   D: bottom left
        ret, p1 = self._get_intersect(A, B, D, C)
        if not ret: return False
        x1, y1 = p1

        ret, p2 = self._get_intersect(A, D, B, C)
        if not ret: return False
        x2, y2 = p2

        cv.line(self.original_img, (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])), RED_COLOR, THICKNESS)
        
        m = (y2 - y1) / (x2 - x1)
        c = y1 - m * x1
        
        return m, c
    
    def _construct_modelfield_img(self):
        s_total = {}

        for row in range(self.sample_inc_h, self.grid_res_h, self.sample_inc_h):
            for col in range(self.sample_inc_w, self.grid_res_w, self.sample_inc_w):
                q = (col, row)

                if col < (self.grid_res_w / 2):
                    horizon = self.lL_horizon
                    hcam = self.lhcam
                    H = self.lH
                else:
                    horizon = self.rL_horizon
                    hcam = self.rhcam
                    H = self.rH

                q_img = cv.perspectiveTransform(np.array([[q]], np.float32), H)
                q_img = tuple(q_img.squeeze())
                q_img = (int(q_img[0]), int(q_img[1]))

                # equation (3.3)
                d = self._min_dist_line_point(horizon, q_img)
                BB_height = (d * T_PLAYER_CM) / hcam

                BB_width = BB_height * PLAYER_ASPECT_RATIO
                tl = (int(q_img[0] - BB_width // 2), int(q_img[1] - BB_height))
                br = (int(q_img[0] + BB_width // 2), int(q_img[1]))

                B = BoundingBox(tl, br)
                a = self.original_img_without_BBs[tl[1]:int(
                    tl[1]+BB_height), tl[0]:int(tl[0]+BB_width)]

                s = Particle(q, q_img, B, a)

                s_total[q_img] = s

                B.draw(self.original_img)
                cv.imwrite(f"BBs/{row}_{col}.png", a)
                self.grid = cv.circle(self.grid, q, 2, BLUE_COLOR, cv.FILLED)
                self.original_img = cv.circle(
                    self.original_img, q_img, THICKNESS, BLUE_COLOR, cv.FILLED)

        return s_total

    def _get_particles(self):
        return self.s

    def _save_particles(self):
      with open('particles.pkl', 'wb') as f:
        pickle.dump(self.s, f)
        f.close()
