import sys
import cv2 as cv
import numpy as np
from PlayerTracker.player_tracker import PlayerTracker

if __name__ == "__main__":

  np.seterr(divide='ignore', invalid='ignore')
  
  if len(sys.argv) != 4: sys.exit(-1)

  _, lcap_fn, mcap_fn, rcap_fn = sys.argv
  lcap = cv.VideoCapture(lcap_fn)
  mcap = cv.VideoCapture(mcap_fn)
  rcap = cv.VideoCapture(rcap_fn)

  mf_gui_clicks = [ (311, 110), (616, 101), (922, 103), # the three top corners
                    (1196, 223), (619, 265), (27, 240), # the three bottom corners
                    (195, 162), (193, 142), # the left post corners
                    (1035, 152), (1037, 132) ] # the right post corners
  
  tracker = PlayerTracker(lcap, mcap, rcap, 
  mf_enable=True, pd_enable=True, bg_enable=False, save_pd=True,
  saved_frames_no=120*25, samples_per_meter=3,
  clicks=mf_gui_clicks)

  tracker.run()
    
  lcap.release()
  rcap.release()
  mcap.release()
