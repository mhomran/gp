from PyQt5 import QtWidgets
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
    self.setWindowTitle("Trackista")

    self.init_ui()

    self.input = input

  def init_ui(self):
    self.Title = QtWidgets.QLabel(self)
    self.Title.setText("Please enter the camera feed")
    self.Title.move(50, 20)
    custom_font = QFont()
    custom_font.setWeight(80)
    custom_font.setPixelSize(30)
    QApplication.setFont(custom_font, "QLabel")
    self.Title.setFont(custom_font)
    self.Title.adjustSize()
  
    self.pick_left = QtWidgets.QPushButton(self)
    self.pick_left.setText("browse to the left feed")
    self.pick_left.clicked.connect(self.pick_left_event)
    custom_font.setPixelSize(15)
    self.pick_left.setFont(custom_font)
    self.pick_left.move(50, 100)
    self.pick_left.adjustSize()

    self.pick_mid = QtWidgets.QPushButton(self)
    self.pick_mid.setText("browse to the mid feed")
    self.pick_mid.clicked.connect(self.pick_mid_event)
    custom_font.setPixelSize(15)
    self.pick_mid.setFont(custom_font)
    self.pick_mid.move(50, 150)
    self.pick_mid.adjustSize()

    self.pick_right = QtWidgets.QPushButton(self)
    self.pick_right.setText("browse to the right feed")
    self.pick_right.clicked.connect(self.pick_right_event)
    custom_font.setPixelSize(15)
    self.pick_right.setFont(custom_font)
    self.pick_right.move(50, 200)
    self.pick_right.adjustSize()

    self.run_btn = QtWidgets.QPushButton(self)
    self.run_btn.setText("start")
    self.run_btn.clicked.connect(self.run_app)
    custom_font.setPixelSize(15)
    self.run_btn.setFont(custom_font)
    self.run_btn.move(50, 300)
    self.run_btn.adjustSize()

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
