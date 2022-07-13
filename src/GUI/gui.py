from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QFont

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

    if not self.input.get_lcap():
      print("please choose the left feed")
      ret = False

    if not self.input.get_mcap():
      print("please choose the mid feed")
      ret = False

    if not self.input.get_rcap():
      print("please choose the right feed")
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

  def set_state(self, state):
    self.state = state
