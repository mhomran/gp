import cv2 as cv
import numpy as np
import math

class Undistorter:
  def __init__(self, k1):
    self.k1 = k1

    self.img_gpu = cv.cuda_GpuMat()
    self.map_x_gpu = cv.cuda_GpuMat()
    self.map_y_gpu = cv.cuda_GpuMat()

  def undistort(self, img_gpu):
    if not self.map_x_gpu.empty() and not self.map_y_gpu.empty():
      new_img = cv.cuda.remap(img_gpu, self.map_x_gpu, self.map_y_gpu, cv.INTER_LINEAR)
      return new_img

    r = lambda x_d, y_d, c_x, c_y: math.sqrt(((x_d - c_x) ** 2) + ((y_d - c_y) ** 2))
    x_img = lambda x_d, c_x, r, k_1: c_x + (1 + k_1*r)*(x_d - c_x)
    y_img = lambda y_d, c_y, r, k_1: c_y + (1 + k_1*r)*(y_d - c_y)

    w, h = img_gpu.size()
    c_x = w//2
    c_y = h//2
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)  
    for x_d in range(w):
      for y_d in range(h):
        new_x = x_img(x_d, c_x, r(x_d, y_d, c_x, c_y), self.k1)
        new_y = y_img(y_d, c_y, r(x_d, y_d, c_x, c_y), self.k1)
        
        map_x[y_d, x_d] = new_x
        map_y[y_d, x_d] = new_y
    
    self.map_x_gpu.upload(map_x)
    self.map_y_gpu.upload(map_y)

    new_img = cv.cuda.remap(img_gpu, self.map_x_gpu, self.map_y_gpu, cv.INTER_LINEAR)
    return new_img
