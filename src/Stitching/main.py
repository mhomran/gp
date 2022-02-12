import cv2 as cv
import numpy as np
import time
import imutils
from stitcher import *
from undistorter import Undistorter
from player_tracker import PlayerTracker


def preprocess(colored_src, mask=None):
  if mask is not None:
    colored_src[mask != 255] = 0
    return colored_src

  r_channel = np.array(colored_src[:, :, 0], dtype=np.int16)
  g_channel = np.array(colored_src[:, :, 1], dtype=np.int16)
  b_channel = np.array(colored_src[:, :, 2], dtype=np.int16)
  dom_r = r_channel > g_channel + 2
  dom_b = b_channel > g_channel + 2
  low_g = g_channel < .2 * 255

  segmented = np.ones_like(r_channel, dtype=np.uint8) * 255
  segmented[dom_r] = 0
  segmented[dom_b] = 0
  segmented[low_g] = 0

  contours, _ = cv.findContours(
      segmented, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
  contours = list(contours)
  contours.sort(key=cv.contourArea, reverse=True)

  mask = np.zeros_like(colored_src[:, :, 0], dtype=np.uint8)
  cv.drawContours(mask, contours, 0, 255, -1)
  colored_src[mask != 255] = 0

  # cv.imshow("seg", cv.resize(segmented, (960, 640)))
  # cv.imshow("mask", cv.resize(mask, (960, 640)))

  return colored_src, mask


result = None
out = None
out_w = None
out_h = None

start_time = time.time()
if __name__ == "__main__":

  np.seterr(divide='ignore', invalid='ignore')

  lcap = cv.VideoCapture('../../data/videos/NewCam/L.mp4')
  mcap = cv.VideoCapture('../../data/videos/NewCam/C.mp4')
  rcap = cv.VideoCapture('../../data/videos/NewCam/R.mp4')

  tracker = PlayerTracker(lcap, mcap, rcap)
  tracker.run()
    
  lcap.release()
  rcap.release()
  mcap.release()
