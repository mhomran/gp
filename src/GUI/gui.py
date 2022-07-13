from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFont
import re

class PlayerTrackerWin(QMainWindow):
  def __init__(self, input) -> None:
    super(PlayerTrackerWin, self).__init__()
    xpos = 500
    ypos = 500
    width = 600
    height = 450

    self.setGeometry(xpos, ypos, width, height)

    uic.loadUi("GUI/design.ui", self)
    self.init_ui()
    self.setWindowTitle("Trackista")

    self.input = input

  def init_ui(self):
    self.pick_left_btn = self.findChild(QtWidgets.QPushButton, "pick_left_btn")
    self.pick_mid_btn = self.findChild(QtWidgets.QPushButton, "pick_mid_btn")
    self.pick_right_btn = self.findChild(QtWidgets.QPushButton, "pick_right_btn")
    self.run_btn = self.findChild(QtWidgets.QPushButton, "run_btn")
    self.start_txt = self.findChild(QtWidgets.QTextEdit, "start_txt")
    self.end_txt = self.findChild(QtWidgets.QTextEdit, "end_txt")
    self.learning_frames_txt = self.findChild(QtWidgets.QTextEdit, "learning_frames_txt")

    self.pick_left_btn.clicked.connect(self.pick_left_event)
    self.pick_mid_btn.clicked.connect(self.pick_mid_event)
    self.pick_right_btn.clicked.connect(self.pick_right_event)
    self.run_btn.clicked.connect(self.run_app)

  def pick_left_event(self):
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 
    'Pick the left feed', "../data" , 'Video Files (*.avi *.mp4)'
    , options=options)
    if file_path:
      self.input.set_lcap(file_path)
  
  def pick_mid_event(self):
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 
    'Pick the mid feed', "../data" , 'Video Files (*.avi *.mp4)'
    , options=options)

    if file_path:
      self.input.set_mcap(file_path)
  
  def pick_right_event(self):
    options = QtWidgets.QFileDialog.Options()
    options |= QtWidgets.QFileDialog.DontUseNativeDialog
    file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 
    'Pick the right feed', "../data" , 'Video Files (*.avi *.mp4)', 
    options=options)
    if file_path:
      self.input.set_rcap(file_path)

  def run_app(self):
    ret = True
    txt = None
    regex = None
    learning_frames = None

    if not self.input.get_lcap():
      # TODO: alert
      print("please choose the left feed")
      ret = False

    if not self.input.get_mcap():
      print("please choose the mid feed")
      ret = False

    if not self.input.get_rcap():
      print("please choose the right feed")
      ret = False

    regex = "^([1-5][0-9]|0?[0-9]):([1-5][0-9]|0?[0-9])$"

    txt = self.start_txt.toPlainText()
    match = re.search(regex, txt)
    if not match:
      print("please enter the start")
      ret = False

    txt = self.end_txt.toPlainText()
    match = re.search(regex, txt)
    if not match:
      print("please enter the end")
      ret = False

    learning_frames = self.learning_frames_txt.toPlainText()
    if not learning_frames:
      ret = False
      print("please choose the number of learning frames")
    else:
      try:
        learning_frames = int(learning_frames)
      except:
        print("please choose the number of learning frames")
        ret = False
    
    self.input.set_start(self.start_txt.toPlainText())
    self.input.set_end(self.end_txt.toPlainText())
    self.input.set_learning_frames(learning_frames)


    if self.input.get_start() and self.input.get_end():
      start = self.input.get_start()
      end = self.input.get_end()
      start_m, start_s = start.split(':')
      start_m = int(start_m)
      start_s = int(start_s)
      start = int((start_m*60+start_s))

      end_m, end_s = end.split(':')
      end_m = int(end_m)
      end_s = int(end_s)
      end = int((end_m*60+end_s))
      if end <= start:
        print("Please choose the right start and end.")
        ret = False

    if ret:
      self.input.set_state("success")
      self.close()

  def closeEvent(self, event):
    super(QMainWindow, self).closeEvent(event)

class Input:
  def __init__(self) -> None:
    self.lcap = None
    self.mcap = None
    self.rcap = None
    self.state = None
    self.start = None
    self.end = None

  def validate(self) -> None:
    ret = False
    if self.state == "success":
      if self.lcap and self.mcap and self.rcap: 
        ret = True
    return ret

  def get_caps(self):
    return self.lcap, self.mcap, self.rcap

  def set_lcap(self, lcap):
    self.lcap = lcap

  def set_mcap(self, mcap):
    self.mcap = mcap

  def set_rcap(self, rcap):
    self.rcap = rcap

  def get_lcap(self):
    return self.lcap

  def get_mcap(self):
    return self.mcap

  def get_rcap(self):
    return self.rcap

  def get_start(self):
    return self.start

  def get_end(self):
    return self.end

  def get_learning_frames(self):
    return self.learning_frames

  def set_state(self, state):
    self.state = state

  def set_start(self, start):
    self.start = start

  def set_end(self, end):
    self.end = end
  
  def set_learning_frames(self, learning_frames):
    self.learning_frames = learning_frames