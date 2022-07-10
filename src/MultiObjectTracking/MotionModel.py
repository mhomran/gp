from operator import mod
import numpy as np
import random
class MotionModel:
  def __init__(self, sigma) -> None:
    """
    Description: The motion model evaluates the likelihood of a by simply
    measuring the distance dmotion between the predicted position
    of the track and the location of the particle on the model 
    soccer Field.

    Input:
      - sigma: is the standard deviation of the normal distribution determining
      the interval of the motion likelihood values. Recall the bell shape of a 
      normal distribution and note that choosing a relatively low sgima will 
      result in a more pointy curve and hence, a larger penalty will be applied
      as the distance between the predicted position and the particle location 
      increases.
    """
    self.sigma = sigma

  def _euclidean_dist(self, p1, p2):
    """
    Description: return the euclidean distance between two
    points.
    
    Input:
      - p1 the first point tuple (x, y)
      - p2 the second point tuple (x, y)
    """
    if isinstance(p2,tuple):
      p2 = np.array([[p2[0]],[p2[1]]])
    if isinstance(p1,tuple):
      p1 = np.array([[p1[0]],[p1[1]]])
      
    diff = p1 - p2
    dist = np.sqrt(diff[0][0]**2+diff[1][0]**2)
    return dist

  def calc(self, p, q):
    """
    Description: calculate the likelihood.
    
    Input:
      - p: the predicted position of track x_n.
      - q: the location of s_m
    """
    d = self._euclidean_dist(p, q)
    sigma = self.sigma
    delta = np.exp(-(d**2) / ((self.sigma**2) * 2))# / (self.sigma * np.sqrt(2*np.pi))
    

    # delta = 1/(sigma * np.sqrt(2*np.pi)) *(np.exp((p-q)/(2*sigma)))
    # delta = random.gauss(d,self.sigma)
    return delta

if __name__ == "__main__":
    sigma = 100
    model = MotionModel(sigma)  
    p1= np.array([[150],[150]])
    p2 =np.array([[200],[200]])
    delta = model.calc(p1,p2)
    print(f'[{p1},{p2}]', delta)