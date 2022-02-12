import cv2 as cv
import numpy as np
import time
import imutils
import math

def warpImages(lframe, rframe, H, ref="r"):
  l_rows, l_cols = lframe.shape[:2]
  r_rows, r_cols = rframe.shape[:2]

  r_corners = np.float32([[0,0], [0, r_rows], [r_cols, r_rows], [r_cols, 0]]).reshape(-1, 1, 2)
  l_corners = np.float32([[0,0], [0, l_rows], [l_cols, l_rows], [l_cols, 0]]).reshape(-1, 1, 2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  tr_corners = r_corners if ref=="l" else l_corners
  tr_corners = cv.perspectiveTransform(tr_corners, H)

  ref_corners = l_corners if ref=="l" else r_corners
  total_corners = np.concatenate((ref_corners, tr_corners), axis=0)

  [x_min, y_min] = np.int32(total_corners.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(total_corners.max(axis=0).ravel() + 0.5)
  
  
  if ref == 'l':
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    output_img = cv.warpPerspective(rframe, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:l_rows+translation_dist[1], translation_dist[0]:l_cols+translation_dist[0]] = lframe
  else:
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])
    output_img = cv.warpPerspective(lframe, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:r_rows+translation_dist[1], translation_dist[0]:r_cols+translation_dist[0]] = rframe

  return output_img

def stitch(frames, M=None, ref="r"):
  lframe, rframe = frames

  if M is not None:
    result = warpImages(lframe, rframe, M, ref)
    return result

  # Create our ORB detector and detect keypoints and descriptors
  orb = cv.ORB_create(nfeatures=2000)

  # Find the key points and descriptors with ORB
  lframe_kp, lframe_desc = orb.detectAndCompute(lframe, None)
  rframe_kp, rframe_desc = orb.detectAndCompute(rframe, None)

  # Create a BFMatcher object.
  # It will find all of the matching keypoints on two images
  bf = cv.BFMatcher_create(cv.NORM_HAMMING)

  # Find matching points
  matches = []
  if ref == 'r':
    matches = bf.knnMatch(lframe_desc, rframe_desc, k=2)
  else:
    matches = bf.knnMatch(rframe_desc, lframe_desc, k=2)
  
  good = []
  for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good.append(m)

  # Set minimum match condition
  MIN_MATCH_COUNT = 10

  if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = None
    dst_pts = None
    if ref == "r":
      src_pts = np.float32([ lframe_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
      dst_pts = np.float32([ rframe_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    else:
      src_pts = np.float32([ rframe_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
      dst_pts = np.float32([ lframe_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Establish a homography
    M, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    
    result = warpImages(lframe, rframe, M, ref)

    return result, M

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

M1 = M2 = M3 = None # homography matrices
mask = None # Soccer field mask
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

  lcap = cv.VideoCapture('../../data/videos/L.mp4')
  mcap = cv.VideoCapture('../../data/videos/C.mp4')
  rcap = cv.VideoCapture('../../data/videos/R.mp4')

  if (lcap.isOpened()== False) or (rcap.isOpened()== False) or (mcap.isOpened()== False): 
    raise("Error opening video streams or files")

  lret, lframe = lcap.read()
  h, w, _ = lframe.shape

  fps = lcap.get(cv.CAP_PROP_FPS)
  
  index = 0
  lcap.set(cv.CAP_PROP_POS_FRAMES, index)
  rcap.set(cv.CAP_PROP_POS_FRAMES, index)
  mcap.set(cv.CAP_PROP_POS_FRAMES, index)

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

      # homography
      if M1 is None or M2 is None or M3 is None:
        lm_res, M1 = stitch([lframe, mframe], ref="r")
        # cv.imwrite("lm_res.png", lm_res)
        mr_res, M2 = stitch([mframe, rframe], ref="l")
        # cv.imwrite("mr_res.png", mr_res)
        result, M3 = stitch([lm_res, mr_res], ref="l")
      else:
        lm_res = stitch([lframe, mframe], M=M1, ref="r")
        mr_res = stitch([mframe, rframe], M=M2, ref="l")
        result = stitch([lm_res, mr_res], M=M3, ref="l")

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
