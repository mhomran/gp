import cv2 as cv
import sys
import imutils

def main(testcase):
    lframe = cv.imread(f'{sys.path[0]}/testcases/{testcase}/2.png')
    rframe = cv.imread(f'{sys.path[0]}/testcases/{testcase}/3.png')

    sift = cv.SIFT_create()
    lframe_kp, lframe_desc = sift.detectAndCompute(lframe, None)
    rframe_kp, rframe_desc = sift.detectAndCompute(rframe, None)

    brute_force_matcher = cv.BFMatcher()
    matched_points = brute_force_matcher.knnMatch(lframe_desc, rframe_desc, k=2)

    sift_good_points = []
    for first_closest, second_closest in matched_points:
      if first_closest.distance < 0.75*second_closest.distance:
        sift_good_points.append(first_closest)

    matching_points = cv.drawMatches(lframe, lframe_kp, rframe, rframe_kp, sift_good_points, None)
    matching_points = imutils.resize(matching_points, 1700)
    cv.imshow("matching points", matching_points)

    cv.waitKey(0)

if __name__ == "__main__":
    if len(sys.argv) != 2: 
        print("[ERROR] No command line input")
        sys.exit(-1)

    main(sys.argv[1])