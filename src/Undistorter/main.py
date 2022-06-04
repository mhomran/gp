import cv2
import numpy as np
import math
from scipy import stats
import imutils
from undistorter import *
import matplotlib.pyplot as plt


# TODO: Configure
IMG_PATH = "L0.jpg"
LINES_NO = 3
POINTS_PER_LINE = 10

# Constants
GUI_WIDTH = 1200

points = np.zeros((POINTS_PER_LINE, 2),int)
points_in_gui = np.zeros((POINTS_PER_LINE, 2),int)
lines = np.zeros((LINES_NO, POINTS_PER_LINE, 2),int)
currentline = 0
counter = 0

def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        points_in_gui[counter%POINTS_PER_LINE] = (x, y)
        points[counter%POINTS_PER_LINE] = _gui2orig(original_img, gui_img, (x, y))
        counter +=1

def _gui2orig(original_img, gui_img, p):
    x = p[0] * original_img.shape[1] // gui_img.shape[1]
    y = p[1] * original_img.shape[0] // gui_img.shape[0] 
    return (x, y)

original_img = cv2.imread(IMG_PATH)
h, w = original_img.shape[:2]
original_img = imutils.resize(original_img, width=w//2)

# Get points from the GUI
for i in range(LINES_NO):
    counter = 0
    gui_img = original_img.copy()
    gui_img = imutils.resize(gui_img, width=GUI_WIDTH)
    points = np.zeros((POINTS_PER_LINE, 2),int)
    points_in_gui = np.zeros((POINTS_PER_LINE, 2),int)
    while True:
        for point in points_in_gui:
            cv2.circle(gui_img, point, 3, (255,0,0), cv2.FILLED)
        cv2.imshow("GUI", gui_img)
        cv2.setMouseCallback("GUI", mousePoints)
        cv2.waitKey(1)
        if counter == POINTS_PER_LINE:
            break
    lines[currentline] = points
    currentline += 1
cv2.destroyAllWindows()

# calculate best K
r = lambda x_d, y_d, c_x, c_y: math.sqrt(((x_d - c_x) ** 2) + ((y_d - c_y) ** 2))
x_img = lambda x_d, c_x, r, k_1: c_x + (1 + k_1*r)*(x_d - c_x)
y_img = lambda y_d, c_y, r, k_1: c_y + (1 + k_1*r)*(y_d - c_y)
k1s =[]
errors = []
h, w = original_img.shape[:2]
c_x = w//2
c_y = h//2
points =[]
lowest_error = float('inf')
bestk1 = float('inf')
best_new_x = []
best_new_y = []

for k1 in range(int(-1e4), int(1e4), 1):
    k1 = k1/1e6
    k1s.append(k1)
    error = 0
    all_new_x = []
    all_new_y = []

    for line in lines:
        
        new_xs =[]
        new_ys =[]
        for point in line:
            x = point[0]
            y = point[1]
            
            new_x = x_img(x, c_x, r(x, y, c_x, c_y), k1)
            new_y = y_img(y, c_x, r(x, y, c_x, c_y), k1)
            new_xs.append(new_x)
            new_ys.append(new_y)
            all_new_x.append(new_x)
            all_new_y.append(new_y)

        # get the line equation
        slope, intercept, r_value, p_value, std_err = stats.linregress(new_xs,new_ys)
        for x,y in zip(new_xs,new_ys):
            error += (x*slope-y+intercept)**2/math.sqrt(1+slope**2)

    if error < lowest_error:
        lowest_error = error
        bestk1 = k1
        best_new_x = all_new_x
        best_new_y = all_new_y


# visualization
print(f"best K: {bestk1}")

img = original_img
myundistorter = Undistorter(img, bestk1) 
img_gpu = cv.cuda_GpuMat()
img_gpu.upload(img)
img_gpu = myundistorter.undistort(img_gpu)
img = img_gpu.download()

plt.scatter(best_new_x, best_new_y)
plt.show()

cv2.imwrite(f"{IMG_PATH}_undistorted.png", img)
