import cv2 as cv
import numpy as np
import time
import imutils
from stitcher import *
from undistorter import Undistorter

# undistortion parameter
lk1 = -.00008 # the higher, the lesser Barrel distortion
mk1 = -.00005 
rk1 = -.00005  

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

  l_undistorter = Undistorter(lk1)
  m_undistorter = Undistorter(mk1)
  r_undistorter = Undistorter(rk1)
  
  lframe_gpu = cv.cuda_GpuMat()
  mframe_gpu = cv.cuda_GpuMat()
  rframe_gpu = cv.cuda_GpuMat()

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
      # upload on GPU
      lframe_gpu.upload(lframe)
      mframe_gpu.upload(mframe)
      rframe_gpu.upload(rframe)      

      # resize
      h, w, _ = lframe.shape
      nw = w//2
      nh = int(h/w*nw)
      lframe_gpu = cv.cuda.resize(lframe_gpu, (nw, nh))
      mframe_gpu = cv.cuda.resize(mframe_gpu, (nw, nh))
      rframe_gpu = cv.cuda.resize(rframe_gpu, (nw, nh))

      # undistort
      lframe_gpu = l_undistorter.undistort(lframe_gpu)
      mframe_gpu = m_undistorter.undistort(mframe_gpu)
      rframe_gpu = r_undistorter.undistort(rframe_gpu)
      lframe = lframe_gpu.download()
      mframe = mframe_gpu.download()
      rframe = rframe_gpu.download()

      lm_res, lm_res_gpu = lm_stitcher.stitch(lframe, mframe, lframe_gpu, mframe_gpu)
      mr_res, mr_res_gpu = mr_stitcher.stitch(mframe, rframe, mframe_gpu, rframe_gpu)
      result, _ = lmr_stitcher.stitch(lm_res, mr_res, lm_res_gpu, mr_res_gpu)

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

      # out.write(result)
      cv.imshow("result", imutils.resize(result, width=1200))
      cv.waitKey(1)
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
