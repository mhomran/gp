import sys
import cv2 as cv
import numpy as np
from PlayerTracker.player_tracker import PlayerTracker

if __name__ == "__main__":

  np.seterr(divide='ignore', invalid='ignore')
  
  if len(sys.argv) != 8: 
    print("[DEV] unsuffcient commands from GUI")
    sys.exit(-1)

  _, lcap_fn, mcap_fn, rcap_fn, start, end, lf, force_mf = sys.argv
  lcap = cv.VideoCapture(lcap_fn)
  mcap = cv.VideoCapture(mcap_fn)
  rcap = cv.VideoCapture(rcap_fn)

  fps = lcap.get(cv.CAP_PROP_FPS)

  start_m, start_s = start.split(':')
  start_m = int(start_m)
  start_s = int(start_s)
  start = int((start_m*60+start_s)*fps)

  end_m, end_s = end.split(':')
  end_m = int(end_m)
  end_s = int(end_s)
  end = int((end_m*60+end_s)*fps)

  lf = int(lf)
  force_mf = force_mf == 'True'

  tracker = PlayerTracker(lcap, mcap, rcap, start, end, lf,
  mf_enable=True, pd_enable=True, bg_enable=False, save_pd=True,
  samples_per_meter=3, force_mf=force_mf)
  
  tracker.run()

  lcap.release()
  rcap.release()
  mcap.release()
