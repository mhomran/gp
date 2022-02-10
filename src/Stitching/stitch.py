import cv2 as cv
import numpy as np
import time
import imutils
import math
from stitcher import *

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

def undistort(img, k1, map_x=None, map_y=None):
  if map_x is not None and map_y is not None:
    new_img = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)
    return new_img

  r = lambda x_d, y_d, c_x, c_y: math.sqrt(((x_d - c_x) ** 2) + ((y_d - c_y) ** 2))
  x_img = lambda x_d, c_x, r, k_1: c_x + (1 + k_1*r)*(x_d - c_x)
  y_img = lambda y_d, c_y, r, k_1: c_y + (1 + k_1*r)*(y_d - c_y)

  h, w, _ = img.shape
  c_x = w//2
  c_y = h//2
  new_img = np.zeros_like(img)
  map_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
  map_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)  
  for x_d in range(w):
    for y_d in range(h):
      new_x = x_img(x_d, c_x, r(x_d, y_d, c_x, c_y), k1)
      new_y = y_img(y_d, c_y, r(x_d, y_d, c_x, c_y), k1)
      
      map_x[y_d, x_d] = new_x
      map_y[y_d, x_d] = new_y
  
  new_img = cv.remap(img, map_x, map_y, cv.INTER_LINEAR)

  return new_img, map_x, map_y

rmap_x = rmap_y = None # undistortion matrices
lmap_x = lmap_y = None # undistortion matrices
mmap_x = mmap_y = None # undistortion matrices
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

  if (lcap.isOpened()== False) or (rcap.isOpened()== False) or (mcap.isOpened()== False): 
    raise("Error opening video streams or files")

  lret, lframe = lcap.read()
  h, w, _ = lframe.shape

  fps = lcap.get(cv.CAP_PROP_FPS)
  
  index = 0
  lcap.set(cv.CAP_PROP_POS_FRAMES, index)
  rcap.set(cv.CAP_PROP_POS_FRAMES, index)
  mcap.set(cv.CAP_PROP_POS_FRAMES, index)

  lm_stitcher = Stitcher("r")
  mr_stitcher = Stitcher("l")
  lmr_stitcher = Stitcher("l")
  
  frame_count = 0
  prev_seconds = 0
  # Read until video is completed
  while(lcap.isOpened()):
    # Capture frame-by-frame
    lret, lframe = lcap.read()
    rret, rframe = rcap.read()
    mret, mframe = mcap.read()

    frame_count += 1
    
    if lret == True and rret == True and mret == True:

      # resize
      h, w, _ = lframe.shape
      lframe = imutils.resize(lframe, width=w//2)
      mframe = imutils.resize(mframe, width=w//2)
      rframe = imutils.resize(rframe, width=w//2)

      # undistort
      lk1 = -.00008 # the higher, the lesser Barrel distortion
      mk1 = -.00005 
      rk1 = -.00005  
      if lmap_x is None:
        lframe, lmap_x, lmap_y = undistort(lframe, lk1)
        mframe, mmap_x, mmap_y = undistort(mframe, mk1)
        rframe, rmap_x, rmap_y = undistort(rframe, rk1)
      else:
        lframe = undistort(lframe, lk1, map_x=lmap_x, map_y=lmap_y)
        mframe = undistort(mframe, mk1, map_x=mmap_x, map_y=mmap_y)
        rframe = undistort(rframe, rk1, map_x=rmap_x, map_y=rmap_y)

      # cv.imwrite("lud.png", lframe)
      # cv.imwrite("mud.png", mframe)
      # cv.imwrite("rud.png", rframe)

      lm_res = lm_stitcher.stitch(lframe, mframe)
      mr_res = mr_stitcher.stitch(mframe, rframe)
      result = lmr_stitcher.stitch(lm_res, mr_res)

      # cv.imwrite("stitched.png", result)

      # # preprocess
      # if mask is None:
      #   result, mask = preprocess(result)
      # else:
      #   result = preprocess(result, mask=mask)

      # write frame
      if out is None:
        out_w = result.shape[1]
        out_h = result.shape[0]
        out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (out_w, out_h))

      out.write(result)
      # cv.imwrite("result.png", result)
      # break

      seconds = frame_count // fps
      minutes = seconds // 60
      if prev_seconds != seconds:
        end_time = time.time()
        prev_seconds = seconds
        print(seconds, "s")

        print("one second is processed in: ", end_time - start_time)
        start_time = time.time()
      # if seconds == 1:
      #   print("one second elapsed")
      #   break
        
    # Break the loop
    else: 
      break

  lcap.release()
  rcap.release()
  mcap.release()
  out.release()
