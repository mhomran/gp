from PlayerDetection.ImageClass import ImageClass
from PlayerDetection.PlayerDetection import PlayerDetection
from Stitcher.stitcher import Stitcher
from Undistorter.undistorter import Undistorter
from ModelField.model_field import GUI_WIDTH, ModelField
from MultiObjectTracking.object_tracking import PlayerTracking
from Canvas.canvas import Canvas
from Statistics.statistics import Statistics
import cv2 as cv
import imutils
import numpy as np
import pickle
import os

class PlayerTracker:
  # undistortion parameter
  lk1 = 5e-06 
  mk1 = -4.1e-05
  rk1 = 1.5e-05

  RESIZE_FACTOR = 2

  GUI_WIDTH = 1200

  def __init__(self, lcap, mcap, rcap, start, end, learning_frames,
  bg_enable=False, mf_enable=True, pd_enable=True, save_pd=False, 
  samples_per_meter=3, pd_frame_no=300, clicks=None,
  bg_history=500, bg_th=16, bg_limit=1000, force_mf=True,
  output_folder='.'):

    duration = lcap.get(cv.CAP_PROP_FRAME_COUNT)
    if start >= duration or end >= duration:
      raise ValueError("[FATAL] the start or end is longer than the videos durations.")

    if start < learning_frames:
      raise ValueError("[FATAL] the start is not sufficient for GMM learning.")

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
    lmrframe, _ = self.lmr_stitcher.stitch(lmframe_gpu, mrframe_gpu)
    out_h, out_w = lmrframe.shape[:2]
    out_h = out_h - out_h%2
    out_w = out_w - out_w%2
    lmrframe = lmrframe[:out_h, :out_w, :]

    self.saved_frames_no = end - start 
    self.learning_frames = learning_frames
    lcap.set(cv.CAP_PROP_POS_FRAMES, start-self.learning_frames)
    mcap.set(cv.CAP_PROP_POS_FRAMES, start-self.learning_frames)
    rcap.set(cv.CAP_PROP_POS_FRAMES, start-self.learning_frames-2)
    self.fps = lcap.get(cv.CAP_PROP_FPS)

    # background subtractor
    self.bg_enable = bg_enable
    if self.bg_enable:
      self.bg_model = cv.createBackgroundSubtractorMOG2(
        bg_history, bg_th, True)
      self.bg_limit = bg_limit

    # Initialize a canvas
    top_view_shape =imutils.resize(cv.imread('h.png'), 500).shape
    gui_size = imutils.resize(lmrframe, width=GUI_WIDTH).shape
    self.canvas = Canvas(gui_size,top_view_shape=top_view_shape)

    # Model Field
    self.mf_enable = mf_enable
    if self.mf_enable:
      if os.path.exists('.model_field.pkl') and not force_mf:
        try:
          with open('.model_field.pkl', 'rb') as f:
            MF = pickle.load(f)
        except:
          with open('.model_field.pkl', 'wb') as f:
            MF = ModelField(lmrframe, samples_per_meter, self.canvas, 
            clicks=clicks)
            pickle.dump(MF, f)
            f.close()
      else:
        with open('.model_field.pkl', 'wb') as f:
          MF = ModelField(lmrframe, samples_per_meter, self.canvas, 
          clicks=clicks)
          pickle.dump(MF, f)
          f.close()
    
    # Player detection
    self.pd_enable = pd_enable
    self.frameId = 1
    self.save_pd = save_pd
    self.save_pd_init = False
    self.pd_frame_no = pd_frame_no
    if self.pd_enable and self.mf_enable:
      self.IMG = ImageClass()
      self.PD = PlayerDetection(MF, self.IMG,learning_frames)
    
    # tracker 
    self.player_tracker = PlayerTracking(MF, self.canvas, base_path=output_folder)

    # statistics 
    self.statistics = Statistics(MF, input_folder=output_folder)


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
      _, lmframe_gpu = self.lm_stitcher.stitch(self.lframe_gpu, self.mframe_gpu)
      _, mrframe_gpu = self.mr_stitcher.stitch(self.mframe_gpu, self.rframe_gpu)
      lmrframe, _  = self.lmr_stitcher.stitch(lmframe_gpu, mrframe_gpu)
      out_h, out_w = lmrframe.shape[:2]
      out_h = out_h - out_h%2
      out_w = out_w - out_w%2
      lmrframe = lmrframe[:out_h, :out_w, :]

      # 5- Player Detection
      if self.bg_enable:
        if self.frameId < self.bg_limit:
          self.bg_model.apply(lmrframe)
        elif self.frameId == self.bg_limit:
          bg_img = self.bg_model.getBackgroundImage()
          cv.imwrite("background.png", bg_img)

      if self.pd_enable:
        fgMask = self.PD.subBG(lmrframe, self.frameId)
        
        self.PD.preProcessing(fgMask)
        self.PD.loopOnBB()
        
        lmrframe_masked  = lmrframe.copy()
        lmrframe_masked[fgMask==0] = np.zeros(3)
        

        self.IMG.writeTxt(lmrframe, f'{round(self.frameId/int(self.fps),2)}')

      if self.frameId > self.learning_frames:
        self.PD.displayIMGs
        # 6- player tracking
        q, q_img = self.PD.getOutputPD()
        self.player_tracker.process_step(q_img,lmrframe_masked,lmrframe)

        # 7- Save
        if self.frameId > self.saved_frames_no:
          self.statistics.save_statistics()
          break

      else:
        self.canvas.show_canvas(imutils.resize(lmrframe, width = GUI_WIDTH),
        status='loading....',
        info="Press esc to exit.")
        if cv.waitKey(1) == 27:
          break

      print(f"frame #{self.frameId}")
      self.frameId += 1
