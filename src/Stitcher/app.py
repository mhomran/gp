from stitcher import Stitcher
import cv2 as cv
import sys


def main(testcase):
    lframe = cv.imread(f'{sys.path[0]}/{testcase}/1.png')
    mframe = cv.imread(f'{sys.path[0]}/{testcase}/2.png')
    rframe = cv.imread(f'{sys.path[0]}/{testcase}/3.png')

    if lframe is None or mframe is None or rframe is None:
        print("[ERROR] file doesn't exist.")
        return
        
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

    cv.imshow('lmrframe.png', lmrframe)
    cv.waitKey(0)

if __name__ == "__main__":
    if len(sys.argv) != 2: 
        print("[DEV] unsuffcient commands from GUI")
        sys.exit(-1)

    main(sys.argv[1])
