
import numpy as np
import cv2 as cv

H_RANGE = 180
class AppearanceModel:
  def __init__(self, c_h, c_s, c_v, s_th, v_lth, v_uth) -> None:
    self.c_h = c_h
    self.c_s = c_s
    self.c_v = c_v

    if s_th > 0.1:  # saturation (th > 0.1)
      self.s_th = s_th
    else: 
      self.s_th = 0.1
    self.v_lth = v_lth
    self.v_uth = v_uth
    
    

  def _calc_hsv_histogram(self, img): 
    """
    Description: Calculate the HSV histograms.
    
    Input:
      - img: the image in BGR format.

    Output:
      The HS & V histograms concatenated.
    """
    hf = H_RANGE // self.c_h

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    #img = img[img != (0,0,0)]
    h, s, v = img[:,:,0], img[:,:,1]/255, img[:,:,2]/255
    #h, s, v = img[0], img[1]/255, img[2]/255
    msk = (s > self.s_th) & (v > self.v_lth) & (v < self.v_uth)
    ht = h[msk]
    st = s[msk] 

    ht = ht // hf # 0 -> self.c_h (Number of H-bins)
    st = (st * self.c_s).astype(int) - 1 # 0 -> self.c_s (Number of S-Bins) 
    v = (v * self.c_v).astype(int).flatten()

    hs_hist, _ = np.histogram(ht*self.c_s + st, bins= range(self.c_h*self.c_s + 1))
    v_hist, _ = np.histogram(v, bins= range(self.c_v + 2))
    return np.concatenate((hs_hist/np.sum(hs_hist) , v_hist/np.sum(v_hist)))
 

  def calc(self, a, b):
    """
    Description: Calculate the appearance likelihood.

    Input:
      - a: the appearance model (img) of s_m
      - b: the appearance model (img) of the track.
    
    Output:
      appearance model likelihood
    """
    hist1 = self._calc_hsv_histogram(a).astype('float32')
    hist2 = self._calc_hsv_histogram(b).astype('float32')
    hist1 /= np.sum(hist1)
    hist2 /= np.sum(hist2)
    p = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    return p


if __name__ == "__main__":
  img1 = cv.imread('appearance_model_problem/track5detection12first.jpg')
  img2 = cv.imread('appearance_model_problem/track5detection12second.jpg')
  APPEARANCE_MODEL_C_H = 10
  APPEARANCE_MODEL_C_S = 5
  APPEARANCE_MODEL_C_V = 5
  sensor_model =AppearanceModel(APPEARANCE_MODEL_C_H, APPEARANCE_MODEL_C_S, APPEARANCE_MODEL_C_V, 0.1, 0.2, 0.9)
  result,_ = sensor_model.calc(img1,img2)
  h1, w1 = img1.shape[:2]
  h2, w2 = img2.shape[:2]

  #create empty matrix
  vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

  #combine 2 images
  vis[:h1, :w1,:3] = img1
  vis[:h2, w1:w1+w2,:3] = img2
  cv.imwrite(f'appearance_model_problem/results/1.jpg',vis)
  print(result)