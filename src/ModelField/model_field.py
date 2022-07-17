import cv2 as cv
import numpy as np
from enum import Enum
import imutils
import pickle

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
GUI_WIDTH = 1700


class BoundingBox:
    def __init__(self, tl, br):
        self.tl = tl
        self.br = br
    def getArea(self):
        return (self.br[0] - self.tl[0]) * (self.br[1] - self.tl[1])    

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
    def __init__(self, img, samples_per_meter, canvas, clicks=None):
        self.original_img = img.copy()
        self.original_img_without_BBs = img.copy()

        self.gui_img = img.copy()
        self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)

        # modelfield image (Top view)
        self.grid = cv.imread("h.png")
        self.grid_res_w = self.grid.shape[1]
        self.px_per_m_w = self.grid_res_w // SOCCER_WIDTH_M
        self.sample_inc_w = int(self.px_per_m_w // samples_per_meter)
        self.grid_res_h = self.grid.shape[0]
        self.px_per_m_h = self.grid_res_h // SOCCER_HEIGHT_M
        self.sample_inc_h = int(self.px_per_m_h // samples_per_meter)

        self.clicks = []
        self.s = None  # particles indexed by q_img
        self.s_by_q = None  # particles indexed by q
        self.L_horizon = None  # the horizon line (m, c)
        self.hcam = None  # camera height
        self.H = None  # homography matrix
        self.gui_state = GuiState.STATE_CORNERS
        self.done = False
        self.hint = "choose the upper left corner"

        if not clicks:

            canvas.set_callback(self.click_event)
            while True:
                self.gui_img = imutils.resize(self.original_img, width=GUI_WIDTH)
                canvas.show_canvas(self.gui_img, status=self.hint, info="Press esc to exit.")
                if self.done:
                    cv.waitKey(1)
                    self.final_input()
                    canvas.show_canvas(self.gui_img, status=self.hint, info="Press esc to exit.")
                    cv.waitKey(2000)
                    break
                if cv.waitKey(1) == 27:
                    exit()
        else:
            for click in clicks:
                self.click_event(cv.EVENT_LBUTTONDOWN, click[0], click[1])

    def _write_hint(self, msg):
        self.hint = msg

    def _gui2orig(self, p):
        x = p[0] * self.original_img.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.original_img.shape[0] // self.gui_img.shape[0]
        return (x, y)

    def click_event(self, event, x, y, flags=None, params=None):
        # checking for left mouse clicks
        if event == cv.EVENT_LBUTTONDOWN:
            self.clicks.append(self._gui2orig((x, y)))
            curr_click = self.clicks[-1]
            self.original_img = cv.circle(
                self.original_img, curr_click, 10, RED_COLOR, cv.FILLED)

            if self.gui_state == GuiState.STATE_CORNERS:
                if len(self.clicks) < 6:
                    if len(self.clicks) == 1:
                        self._write_hint("choose the upper center corner")
                    elif len(self.clicks) == 2:
                        self.upper_center = self.clicks[-1]
                        self._write_hint("choose the upper right corner")
                    elif len(self.clicks) == 3:
                        self._write_hint("choose the bottom right corner")
                    elif len(self.clicks) == 4:
                        self._write_hint("choose the bottom center corner")
                    elif len(self.clicks) == 5:
                        self.bottom_center = self.clicks[-1]
                        self._write_hint("choose the bottom left corner")

                elif len(self.clicks) == 6:
                    half_w = self.grid_res_w // 2
                    pts1 = np.float32(self.clicks[:2] + self.clicks[4:])
                    pts2 = np.float32([[0, 0], [half_w, 0],
                                        [half_w, self.grid_res_h], 
                                        [0, self.grid_res_h]])
                    self.lH = cv.getPerspectiveTransform(pts2, pts1)
                    self.lH_inv = np.linalg.inv(self.lH)
                    A, B, C, D = np.int32(pts1)
                    self.lL_horizon = self._calculate_horizon(A, B, C, D)
                    
                    pts1 = np.float32(self.clicks[1:3] + self.clicks[3:5])
                    pts2 = np.float32([[half_w, 0], [self.grid_res_w, 0],
                                        [self.grid_res_w, self.grid_res_h], 
                                        [half_w, self.grid_res_h]])
                    self.rH = cv.getPerspectiveTransform(pts2, pts1)
                    self.rH_inv = np.linalg.inv(self.rH)

                    A, B, C, D = np.int32(pts1)
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
                    self._write_hint("Please wait till modelfield is constructed.")
                    self.done = True
    
    def final_input(self):
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

        self.s, self.s_by_q = self._construct_modelfield_img()

        self._write_hint("Success.")
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
        """
        Description: get the intersection between the two lines
        the first line passes through A, B. Its slope a, constant b.
        the second line passes through C, D. Its slope c, constant d.
        """

        result = None
        ret = False

        v1, v2 = A[0] == B[0], C[0] == D[0]

        if v1 and v2: 
            ret, result = False, None
        elif v1:
            # y = cx + d
            c = (C[1] - D[1])/(C[0] - D[0])
            d = C[1] - c * C[0]
            x = A[0]
            y = c * x + d
            ret, result = True, (x, y)
        elif v2:
            # y = ax + b
            a = (A[1] - B[1])/(A[0] - B[0])
            b = A[1] - a * A[0]
            x = C[0]
            y = a * x + b
            ret, result = True, (x, y)
        else:
            # y = ax + b
            a = (A[1] - B[1])/(A[0] - B[0])
            b = A[1] - a * A[0]

            # y = cx + d
            c = (C[1] - D[1])/(C[0] - D[0])
            d = C[1] - c * C[0]

            # same slope (no intersection) ?
            if a == c: 
                ret, result = False, None
            else:
                # intersection (solve for x)
                x = (d - b) / (a - c)
                y = (a * x + b)
                ret, result = True, (x, y)

        return ret, result

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
        s_by_q_total = {}

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
                s_by_q_total[q] = s

                self.grid = cv.circle(self.grid, q, 2, BLUE_COLOR, cv.FILLED)
                self.original_img = cv.circle(
                    self.original_img, q_img, THICKNESS, BLUE_COLOR, cv.FILLED)

        return s_total, s_by_q_total

    def get_nearest_particle(self, q_img):
        """
        Description: get the particle that's nearest to the point p

        Input:
            - p: point (x, y)
        output:
            - particle: the nearest particle object or None in case it's out of 
            the field.
        """
        x_img, y_img = q_img
        particle = None

        # Which half the point exists in to know which H_inv to use
        H_inv = None
        # y = ax + b
        A, B = self.upper_center, self.bottom_center
        a = (A[1] - B[1])/(A[0] - B[0])
        b = A[1] - a * A[0]
        if (y_img - a * x_img - b) > 0:
            H_inv = self.lH_inv
        else:
            H_inv = self.rH_inv

        q = cv.perspectiveTransform(np.array([[q_img]], np.float32), H_inv)
        q = tuple(q.squeeze())
        x, y = (int(q[0]), int(q[1]))

        n_x, n_y = round(x / self.sample_inc_w), round(y / self.sample_inc_h)
        n_x, n_y = self.sample_inc_w * int(n_x), self.sample_inc_h * int(n_y)

        if (n_x, n_y) in self.s_by_q:
            particle = self.s_by_q[(n_x, n_y)]

        return particle

    def _get_particles(self):
        return self.s

    def _save_particles(self):
        with open('particles.pkl', 'wb') as f:
            pickle.dump(self.s, f)
            f.close()

    def get_dist_in_meters(self, p1, p2):
        """
        Description: get the distance between two particles 
        in meters.

        Input:
            - p1: the first particle
            - p2: the second particle

        output:
            - dist: the distance in meters
        """
        dist = None

        q1 = p1.q
        q2 = p2.q

        w = np.abs(q1[0] - q2[0])
        h = np.abs(q1[1] - q2[1])

        h_in_m = h / self.px_per_m_h 
        w_in_m = w / self.px_per_m_w 
        
        dist = np.sqrt((w_in_m**2)+(h_in_m**2))

        return dist

    def convert_px2m(self, dist):
        """
        Description: convert the distance in pixels to meters.

        Input:
            - dist: a tuple contains the distance in pixels for
            the height and width respectively.

        output:
            - res: a tuple contains the distance in meters for 
            the height and width respectively
        """
        res = None

        h, w = dist
        h_in_m = h / self.px_per_m_h 
        w_in_m = w / self.px_per_m_w 

        return h_in_m, w_in_m


