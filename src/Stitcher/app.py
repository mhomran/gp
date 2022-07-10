from stitcher import Stitcher
import cv2 as cv

lframe = cv.imread('dataset/000102.jpg')
mframe = cv.imread('dataset/000110.jpg')
rframe = cv.imread('dataset/000127.jpg')

cv.imwrite("1.png", lframe)
cv.imwrite("2.png", mframe)
cv.imwrite("3.png", rframe)

lframe_gpu = cv.cuda_GpuMat()
mframe_gpu = cv.cuda_GpuMat()
rframe_gpu = cv.cuda_GpuMat()

lframe_gpu.upload(lframe)
mframe_gpu.upload(mframe)
rframe_gpu.upload(rframe)

lm_stitcher = Stitcher(lframe, mframe, "r")
mr_stitcher = Stitcher(mframe, rframe, "l")
lmframe, lmframe_gpu = lm_stitcher.stitch(lframe_gpu, mframe_gpu)
mrframe, mrframe_gpu = mr_stitcher.stitch(mframe_gpu, rframe_gpu)
lmr_stitcher = Stitcher(lmframe, mrframe, "l")
lmrframe, _ = lmr_stitcher.stitch(lmframe_gpu, mrframe_gpu)
out_h, out_w = lmrframe.shape[:2]
out_h = out_h - out_h%2
out_w = out_w - out_w%2
lmrframe = lmrframe[:out_h, :out_w, :]

cv.imwrite('result.png', lmrframe)
cv.imshow('lmrframe.png', lmrframe)
cv.waitKey(0)
