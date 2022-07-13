# Import python libraries
from tempfile import tempdir
import cv2, imutils
import cv2 as cv
import copy
import csv
# from detectors import Detectors
import numpy as np 
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
        self.paths = [[]]*23
        self.frameId = 1
        self.clicks = []
        self.track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
        self.team_colors = [(255,255,255),(0,0,255),(0,255,255)]

    def _gui2orig(self, p):
        x = p[0] * self.original_frame.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.original_frame.shape[0] // self.gui_img.shape[0]
        return (x, y)
    def _write_hint(self,img, msg, color=(0, 0, 0)):
        cv.rectangle(img, (10, 2), (300, 20), (255, 255, 255), -1)
        cv.putText(img, msg, (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color)  
    def switchPlayers(self,original_frame):
        
        self.done = False
        self._write_hint(original_frame,"choose two players too correct")
        self.gui_img = original_frame.copy()
        self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)
        cv.namedWindow("GUI")
        cv.setMouseCallback('GUI', self.clickEvent)
        while True:
            cv.imshow('GUI', self.gui_img)
            if self.done:
                cv.waitKey(500)
                cv.destroyAllWindows()
                break
            cv.waitKey(1)
        
    def clickEvent(self, event, x, y, flags=None, params=None):
        # checking for left mouse clicks
        if event != cv.EVENT_LBUTTONDOWN : return 
        self.clicks.append(self._gui2orig((x, y)))
        if len(self.clicks)==2:
            track1_id = self.closest_track(self.clicks[0])
            track2_id = self.closest_track(self.clicks[1])
            self.clicks = []
            if track1_id  and track2_id:
                self.swap_tracks(track1_id,track2_id)
            self.done = True
            
    def swap_tracks(self,track1_id,track2_id):
        # swap ids
        self.tracker.tracks[track1_id].track_id = track2_id
        self.tracker.tracks[track2_id].track_id = track1_id
        
        # swap teams
        temp =  self.tracker.tracks[track1_id].team 
        self.tracker.tracks[track1_id].team = self.tracker.tracks[track2_id].team
        self.tracker.tracks[track2_id].team = temp

        #swap places in array
        temp = self.tracker.tracks[track2_id]
        self.tracker.tracks[track2_id] = self.tracker.tracks[track1_id]  
        self.tracker.tracks[track1_id] = temp
        
  
    def closest_track(self,click):
        min_dist = 1000
        selected_track =  None
        click = np.array([[click[0]],[click[1]]])
        
        for track in self.tracker.tracks:
            current_dist = self._euclidean_dist(click,track.trace[-1])
            if min_dist > current_dist:
                min_dist = current_dist
                selected_track = track.track_id
        return selected_track        
                
    def _euclidean_dist(self, p1, p2):
        diff = p1 - p2
        dist = np.sqrt(diff[0][0]**2+diff[1][0]**2)
        return dist

    def process_step(self,centers,frame,original_frame):
        if (len(centers) == 0):return 
        # Track object using Kalman Filter
        self.tracker.Update(centers, frame,original_frame)
        field_image = copy.deepcopy(self.field_image_orginal)
       
        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        for i in range(len(self.tracker.tracks)):
            self.paths[i].append(self.tracker.tracks[i].top_pos)
            if (len(self.tracker.tracks[i].trace) > 1):
                for j in range(len(self.tracker.tracks[i].trace)-1):
                    # Draw trace line
                    x1 = self.tracker.tracks[i].trace[j][0][0]
                    y1 = self.tracker.tracks[i].trace[j][1][0]
                    x2 = self.tracker.tracks[i].trace[j+1][0][0]
                    y2 = self.tracker.tracks[i].trace[j+1][1][0]
                    clr = self.tracker.tracks[i].track_id % 9
                    cv.line(original_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                self.track_colors[clr], 5)
                
            _write_hint(original_frame, str(self.tracker.tracks[i].track_id), self.tracker.tracks[i].trace[-1])
        # small field image
        for i in range(len(self.tracker.tracks)):
            if (len(self.tracker.tracks[i].trace) > 1):
                for j in range(len(self.tracker.tracks[i].trace)-1):
                    clr = self.tracker.tracks[i].team
                    cv.circle(field_image,self.tracker.tracks[i].top_pos, 75, self.team_colors[clr], -1)
                    x_offset = 100
                    if self.tracker.tracks[i].track_id <10:
                        x_offset = 50
                    _write_hint(field_image, str(self.tracker.tracks[i].track_id), 
                    np.array([[self.tracker.tracks[i].top_pos[0]-x_offset],[self.tracker.tracks[i].top_pos[1]+50]]),font = 4)
        self.original_frame = original_frame
        cv.imshow('field',imutils.resize(field_image, width=GUI_WIDTH//3))
        cv.imshow('Tracking', imutils.resize(original_frame, width=GUI_WIDTH))

        if cv.waitKey(10) == 32:
            cv.destroyAllWindows()
            self.switchPlayers(original_frame)
        
    def write_data(self):

        for i,path in enumerate(self.paths):
            with open(f'countours/{i}player.csv','w') as out:
                csv_out=csv.writer(out)
                
                for row in path:
                    csv_out.writerow(row)







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
                cv.circle(frame_with_detections,(int(point[0]),int(point[1])), 10, (0,0,255), -1)
                _write_hint(frame_with_detections, str(i), ([int(point[0])],[int(point[1])]))
            cv.imshow('Original', imutils.resize(frame_with_detections, width=GUI_WIDTH))
            
            ckpt = 2100
            if (frameId > ckpt):
                if cv.waitKey(0) == 27: exit()
            else:
                cv.waitKey(1)

        frameId += 1


if __name__ == "__main__":
    # execute main
    main()