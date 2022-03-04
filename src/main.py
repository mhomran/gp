import cv2 as cv
import numpy as np
import time
from PlayerTracker.player_tracker import PlayerTracker

start_time = time.time()
if __name__ == "__main__":

  np.seterr(divide='ignore', invalid='ignore')

  lcap = cv.VideoCapture('/media/mhomran/Material/Workspaces/GitHub/gp/data/videos/pyrVSmas/L.mp4')
  mcap = cv.VideoCapture('/media/mhomran/Material/Workspaces/GitHub/gp/data/videos/pyrVSmas/C.mp4')
  rcap = cv.VideoCapture('/media/mhomran/Material/Workspaces/GitHub/gp/data/videos/pyrVSmas/R.mp4')

  tracker = PlayerTracker(lcap, mcap, rcap)
  tracker.run()
    
  lcap.release()
  rcap.release()
  mcap.release()
