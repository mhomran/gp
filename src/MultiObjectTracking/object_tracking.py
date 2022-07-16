# Import python libraries
import cv2, imutils
import cv2 as cv
import copy
import csv
# from detectors import Detectors
import numpy as np 
from MultiObjectTracking.helper import _write_hint
from MultiObjectTracking.tracker import Tracker
from MultiObjectTracking.tracker import Track
from MultiObjectTracking.ColorPlayers import ColorPlayers
from MultiObjectTracking.ClusterPlayers import ClustringModule
from Canvas.canvas import Canvas

base_path = "D:/kollea/gradePorject/gp2/kalman_filter_multi_object_tracking/Data/VideoWithTags"
base_path = "D:/kollea/gradePorject/last_version_gp/src/2,3part/progression"

# base_path = "E:/0Senior/0_GP/detection_data1" 

FRAME_OFFSET = 1
GUI_WIDTH = 1700        
TOP_VIEW_WIDTH = 500

class PlayerTracking(object):
    def __init__(self, MF, canvas,base_path ='.'):
        self.tracker = Tracker(MF, canvas,base_path= base_path)
        self.MF = MF
        # first frame to process 
        self.field_image_orginal = cv2.imread('h.png')
        self.paths = [[]]*23
        self.clustrModule = ClustringModule()
        self.frameId = 0
        self.clicks = []
        self.track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]
        self.team_colors = [(255,255,255),(0,0,255),(0,255,255)]
        self.first_frame = True
        self.canvas = canvas
        self.hint = ''
        self.info = ''
    def _gui2orig(self, p):
        x = p[0] * self.original_frame.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.original_frame.shape[0] // self.gui_img.shape[0]
        return (x, y)
        
    def _write_hint(self,img, msg, color=(0, 0, 0)):
        cv.rectangle(img, (10, 2), (300, 20), (255, 255, 255), -1)
        cv.putText(img, msg, (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.25, color)  

    def draw_top_view(self):
        field_image = copy.deepcopy(self.field_image_orginal)
        for i in range(len(self.tracker.tracks)):
            clr = self.tracker.tracks[i].team
            cv.circle(field_image,self.tracker.tracks[i].top_pos, 10, self.team_colors[clr], -1)
            x_offset = 10
            if self.tracker.tracks[i].track_id <10:
                x_offset = 5
            _write_hint(field_image, str(self.tracker.tracks[i].track_id), 
            np.array([[self.tracker.tracks[i].top_pos[0]-x_offset],[self.tracker.tracks[i].top_pos[1]+5]]),font = 0.5)
        return field_image

    def _draw_tracks(self):
        result_frame = copy.deepcopy(self.clean_original_frame)
        for i in range(len(self.tracker.tracks)):
            if (len(self.tracker.tracks[i].trace) > 1):
                for j in range(len(self.tracker.tracks[i].trace)-1):
                    # Draw trace line
                    x1 = self.tracker.tracks[i].trace[j][0][0]
                    y1 = self.tracker.tracks[i].trace[j][1][0]
                    x2 = self.tracker.tracks[i].trace[j+1][0][0]
                    y2 = self.tracker.tracks[i].trace[j+1][1][0]
                    clr = self.tracker.tracks[i].track_id % 9
                    cv.line(result_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                                self.track_colors[clr], 5)
            player_pos = copy.deepcopy(self.tracker.tracks[i].trace[-1])
            player_pos[0][0] += 20
            _write_hint(result_frame, str(self.tracker.tracks[i].track_id),player_pos )
        return result_frame
    def _write_hint(self, msg):
        self.hint = msg
    def _write_info(self,msg):
        self.info = msg
    def modifyTracks(self):
        self.done = False
        
        self.canvas.set_callback(self.clickEvent)
        self.gui_img = self._draw_tracks()
        self._write_hint("modifying tracks")
        self._write_info('''Press c to swap two tracks \n
Press x to correct a track \n
Press enter to continue.''')
        self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)
        field_image = self.draw_top_view()
        self.field_image  = imutils.resize(field_image, TOP_VIEW_WIDTH)
        self.state = 'idle'
        while True:
            
            self.canvas.show_canvas(self.gui_img,top_view = self.field_image, status=self.hint,info = self.info)
            if self.done:
                cv.waitKey(500)
                break
            k = cv.waitKey(10)
            if k==ord('c'):
                self.state = 'swtich'
                self._write_info("choose two players to swap")
            if k==ord('x'):
                self.state = 'repostion'
                self._write_info("choose a player and a point to correct")
            if k== 13:
                self.done = True    
    
    def repostionPlayer(self,track_id,new_point):
        particle = self.MF.get_nearest_particle((new_point[0],new_point[1]))
        x,y = particle.q_img
        top_pos = particle.q
        point = [[x],[y]]
        track_bb = self.tracker._get_BB_as_img(point,self.masked)
        self.tracker.tracks[track_id] = Track(point,track_id,track_bb,self.tracker.tracks[track_id].team,top_pos)
        
    def clickEvent(self, event, x, y, flags=None, params=None):
        # checking for left mouse clicks
        if event != cv.EVENT_LBUTTONDOWN or self.state =='idle': return 
        self.clicks.append(self._gui2orig((x, y)))

        # if 2 click and repostion
        if len(self.clicks)==2 and self.state == 'repostion':
            track1_id = self.closest_track(self.clicks[0])
            if track1_id is not None:
                self.repostionPlayer(track1_id,self.clicks[1])
                
            self.clicks = []
            self._write_hint("modifying tracks")
            self._write_info('''Press c to swap two tracks \n
Press x to correct a track \n
Press enter to continue.''')
            self.state = 'idle'
            self.gui_img = self._draw_tracks()
            self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)
            field_image = self.draw_top_view()
            self.field_image  = imutils.resize(field_image, TOP_VIEW_WIDTH)
        
        # if two clicks ana swtich
        if len(self.clicks)==2 and self.state == 'swtich':
            track1_id = self.closest_track(self.clicks[0])
            track2_id = self.closest_track(self.clicks[1])
            self.clicks = []
            if track1_id is not None and track2_id is not None:
                self.swap_tracks(track1_id,track2_id)  
            self._write_hint("modifying tracks")
            self._write_info('''Press c to swap two tracks \n 
Press x to correct a track \n
Press enter to continue.''')
            self.state = 'idle'
            self.gui_img = self._draw_tracks()
            self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)
            field_image = self.draw_top_view()
            self.field_image  = imutils.resize(field_image, TOP_VIEW_WIDTH)
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

    def process_step(self,centers,masked,original_frame):
        if (len(centers) == 0 ):return 
        
        # Track object using Kalman Filter
        self.tracker.Update(centers, masked,original_frame)
        self.clean_original_frame = copy.deepcopy(original_frame)
        field_image = copy.deepcopy(self.field_image_orginal)
        self.masked = masked

        # For identified object tracks draw tracking line
        # Use various colors to indicate different track_id
        original_frame = self._draw_tracks()
        self.original_frame = original_frame
        if self.first_frame:
            self.first_frame =False
            chooseColors = ColorPlayers(imutils.resize(original_frame, width=GUI_WIDTH),self.canvas)
            teams = chooseColors.run(0,field_image,
            [track.top_pos for track in  self.tracker.tracks ],
            self.clustrModule.getTeamsColors())
            for i,_ in enumerate(self.tracker.tracks):
                self.tracker.tracks[i].team  = teams[i]
        # small field image
        field_image = self.draw_top_view()
        field_image  = imutils.resize(field_image, TOP_VIEW_WIDTH)
        frame = imutils.resize(original_frame, width=GUI_WIDTH)
        self._write_info('''Press esc to exit \n
Press m to modify tracks''')
        self.canvas.show_canvas(frame, top_view=field_image,info = self.info)
        
       
        k =cv.waitKey(10)
        if k == ord('m'):
            self.modifyTracks()
        if k == 27:
            exit()

    def write_data(self):

        for i,path in enumerate(self.paths):
            with open(f'countours/{i}player.csv','w') as out:
                csv_out=csv.writer(out)
                
                for row in path:
                    csv_out.writerow(row)







