import os
from PlayerDetection.ImageClass import ImageClass
from PlayerDetection.PlayerDetection import PlayerDetection
from PlayerDetection.TagWriter import TagWriter
from Stitcher.stitcher import Stitcher
from Undistorter.undistorter import Undistorter
from ModelField.model_field import ModelField
import cv2 as cv
import imutils, time

class PlayerTracker:
  # undistortion parameter
  lk1 = -5e-06 
  mk1 = 4.1e-05
  rk1 = -1.5e-05

  RESIZE_FACTOR = 2

  GUI_WIDTH = 1200

  def __init__(self, lcap, mcap, rcap, 
  save_pd=False, saved_frames_no=1200, samples_per_meter=3
  , pd_frame_no=300):
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
    MF = ModelField(lmrframe, samples_per_meter)
    particles = MF._get_particles()
    MF._save_particles()
    
    # Background
    self.frameId = 0
    self.IMG = ImageClass()
    self.PD = PlayerDetection(particles, self.IMG)
    self.save_pd = save_pd
    self.saved_frames_no = saved_frames_no
    self.save_pd_init = False
    self.pd_frame_no = pd_frame_no
    self.pd_learned = False

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

      # 5- Player Detection
      fgMask = self.PD.subBG(lmrframe_gpu)
      if self.frameId > self.pd_frame_no:
        self.pd_learned = True
      
      if self.pd_learned:
        self.PD.preProcessing(fgMask)
        self.PD.loopOnBB()

      self.IMG.writeTxt(lmrframe, self.frameId)
      self.IMG.showImage(lmrframe, "Frame")
      keyboard = cv.waitKey(1)
      if keyboard == 'q' or keyboard == 27:
        break

      # 6- Calculate performance for the whole pipeline
      self._calculate_performance()

      # 7- Save
      if self.save_pd and self.pd_learned:
        if not self.save_pd_init:
          self.frameId = 0
          self.lcap.set(cv.CAP_PROP_POS_FRAMES, self.frameId)
          self.mcap.set(cv.CAP_PROP_POS_FRAMES, self.frameId)
          self.rcap.set(cv.CAP_PROP_POS_FRAMES, self.frameId)

          exists = os.path.exists("q")
          if not exists:
            os.makedirs("q")
          exists = os.path.exists("q_img")
          if not exists:
            os.makedirs("q_img")
          self.save_pd_init = True
        else:
          if self.frameId < self.saved_frames_no:
            # save frame
            self.out.write(lmrframe)
            # save tags
            q, q_img = self.PD.getOutputPD()
            TagWriter.write(f"q/{self.frameId}.csv", q)
            TagWriter.write(f"q_img/{self.frameId}.csv", q_img)
          else:
            break
      print(self.frameId)
      self.frameId += 1



  
  