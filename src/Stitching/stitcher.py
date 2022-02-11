import numpy as np
import cv2 as cv

class Stitcher:
  # Set minimum match condition
  MIN_MATCH_COUNT = 10

  def __init__(self, ref):
    self.ref = ref
    self.h = None # Homography Matrix
    self.out_shape = None # stitched image size
    self.trans_dist = None # translation distance
    
  def _warpImages(self, lframe, rframe, lframe_gpu, rframe_gpu):
    l_height, l_width = lframe.shape[:2]
    r_height, r_width = rframe.shape[:2]

    if self.out_shape is None:
      r_corners = np.float32([[0,0], [0, r_height], [r_width, r_height], [r_width, 0]]).reshape(-1, 1, 2)
      l_corners = np.float32([[0,0], [0, l_height], [l_width, l_height], [l_width, 0]]).reshape(-1, 1, 2)

      tr_corners = r_corners if self.ref=="l" else l_corners
      tr_corners = cv.perspectiveTransform(tr_corners, self.h)

      ref_corners = l_corners if self.ref=="l" else r_corners
      total_corners = np.concatenate((ref_corners, tr_corners), axis=0)

      [x_min, y_min] = np.int32(total_corners.min(axis=0).ravel())
      [x_max, y_max] = np.int32(total_corners.max(axis=0).ravel())
      
      self.trans_dist = [-x_min, -y_min]
      self.out_shape = (x_max-x_min, y_max-y_min)

    trans_m = np.array([[1, 0, self.trans_dist[0]], [0, 1, self.trans_dist[1]], [0, 0, 1]])
    if self.ref == 'l':
      output_img_gpu = cv.cuda.warpPerspective(rframe_gpu, trans_m.dot(self.h), self.out_shape)
      output_img = output_img_gpu.download()
      output_img[self.trans_dist[1]:l_height+self.trans_dist[1], self.trans_dist[0]:l_width+self.trans_dist[0]] = lframe
      output_img_gpu.upload(output_img)
    else:
      output_img_gpu = cv.cuda.warpPerspective(lframe_gpu, trans_m.dot(self.h), self.out_shape)
      output_img = output_img_gpu.download()
      output_img[self.trans_dist[1]:r_height+self.trans_dist[1], self.trans_dist[0]:r_width+self.trans_dist[0]] = rframe
      output_img_gpu.upload(output_img)
      
    return output_img, output_img_gpu

  def stitch(self, lframe, rframe, lframe_gpu, rframe_gpu):
    if self.h is None:
      # Initiate SIFT detector
      sift = cv.SIFT_create()

      # find the keypoints and descriptors with SIFT
      lframe_kp, lframe_desc = sift.detectAndCompute(lframe, None)
      rframe_kp, rframe_desc = sift.detectAndCompute(rframe, None)

      # BFMatcher with default params
      bf = cv.BFMatcher()
      matches = bf.knnMatch(lframe_desc, rframe_desc, k=2)

      # Apply ratio test
      good = []
      for m,n in matches:
        if m.distance < 0.75*n.distance:
          good.append(m)

      if len(good) > Stitcher.MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        lframe_kp = np.float32([lframe_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        rframe_kp = np.float32([rframe_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        src_pts = rframe_kp if self.ref == 'l' else lframe_kp
        dst_pts = lframe_kp if self.ref == 'l' else rframe_kp
        
        # Establish a homography
        self.h, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
      
    result = self._warpImages(lframe, rframe, lframe_gpu, rframe_gpu)

    return result
