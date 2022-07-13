# Import python libraries
import cv2, imutils
from cv2 import waitKey
import copy
# from detectors import Detectors
from MultiObjectTracking.helper import _write_hint
from MultiObjectTracking.tracker import Tracker

base_path = "D:/kollea/gradePorject/gp2/kalman_filter_multi_object_tracking/Data/VideoWithTags"
base_path = "D:/kollea/gradePorject/last_version_gp/src/2,3part/progression"
# base_path = "E:/0Senior/0_GP/detection_data1" 

FRAME_OFFSET = 1
GUI_WIDTH = 1500        

class PlayerTracking(object):
    def __init__(self,MF):
        self.tracker = Tracker(MF)
        # first frame to process 
        self.field_image_orginal = cv2.imread('h.png')
        self.frameId = 1
        self.track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
        self.team_colors = [(255,255,255),(0,0,255),(0,255,255)]

    def _gui2orig(self, p,frame):
        x = int(p[0] * self.field_image_orginal.shape[1] // frame.shape[1])
        y = int(p[1] * self.field_image_orginal.shape[0] // frame.shape[0])
        return (x, y)
    def process_step(self,centers,frame,original_frame):
        if (len(centers) == 0):return 
        # Track object using Kalman Filter
        self.tracker.Update(centers, frame,original_frame)
        field_image = copy.deepcopy(self.field_image_orginal)
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(self.tracker.tracks)):
            if (len(self.tracker.tracks[i].trace) > 1):
                for j in range(len(self.tracker.tracks[i].trace)-1):
                    # Draw trace line
                    x1 = self.tracker.tracks[i].trace[j][0][0]
                    y1 = self.tracker.tracks[i].trace[j][1][0]
                    x2 = self.tracker.tracks[i].trace[j+1][0][0]
                    y2 = self.tracker.tracks[i].trace[j+1][1][0]
                    clr = self.tracker.tracks[i].track_id % 9
                    cv2.line(original_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                self.track_colors[clr], 5)
            _write_hint(original_frame, str(self.tracker.tracks[i].track_id), self.tracker.tracks[i].trace[-1])
        for i in range(len(self.tracker.tracks)):
            if (len(self.tracker.tracks[i].trace) > 1):
                for j in range(len(self.tracker.tracks[i].trace)-1):
                    # Draw trace line
                    x1 = self.tracker.tracks[i].trace[j][0][0]
                    y1 = self.tracker.tracks[i].trace[j][1][0]
                    x2 = self.tracker.tracks[i].trace[j+1][0][0]
                    y2 = self.tracker.tracks[i].trace[j+1][1][0]
                    clr = self.tracker.tracks[i].team % 9

                    cv2.circle(field_image,self.tracker.tracks[i].top_pos, 5, self.team_colors[clr], -1)
    
        cv2.imshow('field',imutils.resize(field_image, width=GUI_WIDTH//3))
        cv2.imshow('Tracking', imutils.resize(original_frame, width=GUI_WIDTH))






def main():
    player_tracker = PlayerTracking()

    frameId = 1
    debug = True
    cap = cv2.VideoCapture(f'{base_path}/masked.avi')
    caporiginal = cv2.VideoCapture(f'{base_path}/original.avi')
    while(True):
        # read the frame
        ret, frame = cap.read()
        ret , framoriginal = caporiginal.read()
        # Make copy of original frame
        frame_with_detections = copy.copy(framoriginal)
        # detection phase
        centers = TagReader.read(f"{base_path}/q_img_gt/{FRAME_OFFSET+frameId}.csv")
        # tracking phase
        
        player_tracker.process_step(centers,frame,framoriginal)

        if debug:
            for i, point in enumerate(centers):
                cv2.circle(frame_with_detections,(int(point[0]),int(point[1])), 10, (0,0,255), -1)
                _write_hint(frame_with_detections, str(i), ([int(point[0])],[int(point[1])]))
            cv2.imshow('Original', imutils.resize(frame_with_detections, width=GUI_WIDTH))
            
            ckpt = 2100
            if (frameId > ckpt):
                if cv2.waitKey(0) == 27: exit()
            else:
                cv2.waitKey(1)

        frameId += 1


if __name__ == "__main__":
    # execute main
    main()