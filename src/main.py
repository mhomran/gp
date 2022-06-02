import cv2 as cv
import numpy as np
import time
from PlayerTracker.player_tracker import PlayerTracker

start_time = time.time()
if __name__ == "__main__":

  np.seterr(divide='ignore', invalid='ignore')

  lcap = cv.VideoCapture('../data/videos/pyrVSmas/L.mp4')
  mcap = cv.VideoCapture('../data/videos/pyrVSmas/C.mp4')
  rcap = cv.VideoCapture('../data/videos/pyrVSmas/R.mp4')

  tracker = PlayerTracker(lcap, mcap, rcap, 
  save_pd=True, saved_frames_no=250, samples_per_meter=2)
  tracker.run()
    
  lcap.release()
  rcap.release()
  mcap.release()
