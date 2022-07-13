from PyQt5.QtWidgets import QApplication
from GUI.gui import PlayerTrackerWin, Input
import sys
import numpy as np
from os import system

def gui(input):
  app = QApplication(sys.argv)
  win = PlayerTrackerWin(input)

  win.show()
  ret = app.exec_()
  app.exit()

  return ret


if __name__ == "__main__":
  np.seterr(divide='ignore', invalid='ignore')

  input = Input()

  ret = gui(input)
  
  if not input.validate(): sys.exit(ret)

  lcap, mcap, rcap = input.get_caps()
  start = input.get_start()
  end = input.get_end()
  learning_frames = input.get_learning_frames()



  system(f"python3 main.py {lcap} {mcap} {rcap} {start} {end} {learning_frames}")
