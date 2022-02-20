from Stitcher.stitcher import Stitcher
from Undistorter.undistorter import Undistorter
from ModelField.model_field import ModelField
import cv2 as cv
import imutils

class PlayerTracker:
  # undistortion parameter
  lk1 = -5e-06 
  mk1 = 4.1e-05
  rk1 = -1.5e-05

  RESIZE_FACTOR = 2

  GUI_WIDTH = 1200

  def __init__(self, lcap, mcap, rcap):
    self.lcap = lcap
    self.mcap = mcap
    self.rcap = rcap

    self.lframe_gpu = cv.cuda_GpuMat()
    self.mframe_gpu = cv.cuda_GpuMat()
    self.rframe_gpu = cv.cuda_GpuMat()
    
    # Read frames 
    ret, frames = self._get_frames()
    if not ret: raise "[Tracker]: Can't read frames"
    lframe, mframe, rframe = frames

    # upload them to GPU
    self._upload_images_to_GPU(lframe, mframe, rframe)

    # Resize
    self._resize(PlayerTracker.RESIZE_FACTOR)
    lframe, mframe, rframe = self._download_images_from_GPU()

    # Undistorters
    self.l_undistorter = Undistorter(lframe, PlayerTracker.lk1)
    self.m_undistorter = Undistorter(mframe, PlayerTracker.mk1)
    self.r_undistorter = Undistorter(rframe, PlayerTracker.rk1)
    self._undistort()
    lframe, mframe, rframe = self._download_images_from_GPU()

    # Stitchers
    self.lm_stitcher = Stitcher(lframe, mframe, "r")
    self.mr_stitcher = Stitcher(mframe, rframe, "l")
    lmframe, lmframe_gpu = self.lm_stitcher.stitch(self.lframe_gpu, self.mframe_gpu)
    mrframe, mrframe_gpu = self.mr_stitcher.stitch(self.mframe_gpu, self.rframe_gpu)
    self.lmr_stitcher = Stitcher(lmframe, mrframe, "l")
    lmrframe, lmrframe_gpu = self.lmr_stitcher.stitch(lmframe_gpu, mrframe_gpu)
    
    # Model Field
    ModelField(lmrframe)

    # Background

    # performance
    self.frame_count = 0
    self.start_time = 0
    self.end_time = 0
    self.prev_second = 0

    # Saver
    out_h, out_w = lmrframe.shape[:2]
    self.fps = lcap.get(cv.CAP_PROP_FPS)
    self.out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc('M','J','P','G'), self.fps, (out_w, out_h))
  
  def __del__(self):
    self.out.release()
  
  def _upload_images_to_GPU(self, lframe, mframe, rframe):
    self.lframe_gpu.upload(lframe)
    self.mframe_gpu.upload(mframe)
    self.rframe_gpu.upload(rframe)

  def _download_images_from_GPU(self):
    lframe = self.lframe_gpu.download()
    mframe = self.mframe_gpu.download()
    rframe = self.rframe_gpu.download()
    return [lframe, mframe, rframe]
  
  def _get_frames(self):
    if not self.lcap.isOpened() or not self.mcap.isOpened() or not self.rcap.isOpened():
      raise "[Tracker]: videos aren't opened"
      
    lret, lframe = self.lcap.read()
    rret, rframe = self.rcap.read()
    mret, mframe = self.mcap.read()

    if lret == True and rret == True and mret == True:
      return True, [lframe, mframe, rframe]
    else:
      return False, []
  
  def _resize(self, factor):
    """
    Description: resize the images by reducing their widths with a factor
    """
    w, h = self.lframe_gpu.size()
    nw = w//factor
    nh = int(h/w*nw)
    self.lframe_gpu = cv.cuda.resize(self.lframe_gpu, (nw, nh), interpolation=cv.INTER_AREA)
    self.mframe_gpu = cv.cuda.resize(self.mframe_gpu, (nw, nh), interpolation=cv.INTER_AREA)
    self.rframe_gpu = cv.cuda.resize(self.rframe_gpu, (nw, nh), interpolation=cv.INTER_AREA)

  def _undistort(self):
    self.lframe_gpu = self.l_undistorter.undistort(self.lframe_gpu)
    self.mframe_gpu = self.m_undistorter.undistort(self.mframe_gpu)
    self.rframe_gpu = self.r_undistorter.undistort(self.rframe_gpu)

  def _calculate_performance(self):
    self.frame_count += 1
    curr_second = int(self.frame_count // self.fps)
    if curr_second != self.prev_second:
      self.prev_second = curr_second
      duration = int(self.end_time - self.start_time)
      print(f"Second #{curr_second}")
    # print(f"One second is processed in: {duration} s")

  def _print_images(self):
    lframe, mframe, rframe = self._download_images_from_GPU()
    cv.imshow("lframe", imutils.resize(lframe, width=PlayerTracker.GUI_WIDTH))
    cv.imshow("mframe", imutils.resize(mframe, width=PlayerTracker.GUI_WIDTH))
    cv.imshow("rframe", imutils.resize(rframe, width=PlayerTracker.GUI_WIDTH))
    cv.waitKey(0)
    cv.destroyAllWindows()

  def run(self):

    while self.lcap.isOpened() and self.mcap.isOpened() and self.rcap.isOpened():
      # 1- read frames
      ret, frames = self._get_frames()
      if not ret: return
      lframe, mframe, rframe = frames
      
      # 2- upload to GPU
      self._upload_images_to_GPU(lframe, mframe, rframe)

      # 3- Resize
      self._resize(PlayerTracker.RESIZE_FACTOR)

      # 3- Undistort
      self._undistort()

      # 4- Stitch
      lmframe, lmframe_gpu = self.lm_stitcher.stitch(self.lframe_gpu, self.mframe_gpu)
      mrframe, mrframe_gpu = self.mr_stitcher.stitch(self.mframe_gpu, self.rframe_gpu)
      lmrframe, lmrframe_gpu  = self.lmr_stitcher.stitch(lmframe_gpu, mrframe_gpu)

      # 5- Calculate performance for the whole pipeline
      self._calculate_performance()

      # 6- Save
      self.out.write(lmrframe)


  
  
