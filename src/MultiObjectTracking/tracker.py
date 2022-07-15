# import section 
import numpy as np
from MultiObjectTracking.kalmanfilter import KalmanFilter
import pickle 
from MultiObjectTracking.AppearanceModel import AppearanceModel
from MultiObjectTracking.MotionModel import MotionModel
from ModelField.model_field import ModelField
from MultiObjectTracking.Annotator import Annotator
import cv2 as cv
import copy
import imutils
from enum import Enum
############################
#constants section 
P_MOTION_TH = 0.75
P_COLOR_TH = 0.7
DIST_TH = 100
MAX_FRAME_SKIP = 100
MAX_TRACE_LEN = 5
START_TRACK_ID = 0
GUI_WIDTH = 1500
THICKNESS = 3  # thickness of drawings
# motion model constants
MOTION_MODEL_SGIMA = DIST_TH

# appearance model constants
APPEARANCE_MODEL_C_H = 10
APPEARANCE_MODEL_C_S = 10
APPEARANCE_MODEL_C_V = 10
APPEARANCE_MODEL_S_TH = 0.3
APPEARANCE_MODEL_V_LTH = 0.2
APPEARANCE_MODEL_V_UTH = 1



RED_COLOR = (0, 0, 255)
BLUE_COLOR = (255, 0, 0)
CLICKS  =   [(907, 427,0), #home goal keeper
            (1292, 706,0), (1203, 548,0), (1267, 476,0), (1301, 414,0), (1398, 340,0), (1483, 403,0), (1547, 521,0), (1705, 479,0), (1710, 372,0), (1654, 327,0),
            #home team
            (2815, 407,1),#away goal keeper
            (2264, 590,1), (2201, 434,1), (2261, 383,1), (2110, 320,1), (2074, 421,1), (1986, 396,1), (1912, 450,1), (1723, 574,1), (1708, 418,1), (1801, 338,1),
            #away team
            (1837, 481,2),]#refree

CLICKS = [(858, 425,0),
(1365, 666,0), (1305, 532,0), (1321, 476,0), (1358, 407,0), (1436, 343,0), (1614, 414,0), (1728, 539,0), (1779, 334,0), (2068, 461,0), (2008, 416,0),
(2793, 412,1),
(2315, 603,1), (2350, 452,1), (2304, 374,1), (2119, 309,1), (2137, 427,1), (1983, 394,1), (1832, 441,1), (1672, 320,1), (1472, 434,1), (1467, 592,1),
(1699, 443,2)]

class GuiState(Enum):
    STATE_HOME_GOAL_KEEPER = 1,
    STATE_HOME = 2,
    STATE_AWAY_GOAL_KEEPER = 3,
    STATE_AWAY = 4,
    STATE_REF = 5,
    STATE_END = 6,

#############################
    
#Track class : each player has a track
class Track(object):
    def __init__(self, prediction, trackId, track_bb,team,top_pos):
        self.track_id = trackId  # identification of each track object
        self.KF = KalmanFilter(prediction[0],prediction[1])  # KF instance to track this object
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = [prediction]  # trace path

        self.prediction = np.asarray(prediction)  # predicted centroids (x,y)
        self.track_bb = track_bb

        self.color_group = None
        if team == None: team=7
        self.team = team
        self.top_pos = top_pos

        self.last_measurment = prediction
##########################
    
#Tracker class handles all the tracks all of all the players
class Tracker(object):
    def __init__(self,MF):
        
        self.max_frames_to_skip = MAX_FRAME_SKIP
        self.max_trace_length = MAX_TRACE_LEN
        self.next_trackId = START_TRACK_ID
        self.tracks = []
        self.p_motion = None
        self.p_color = None
        self.init_state = GuiState.STATE_HOME_GOAL_KEEPER
        self.frame_count = 0
        self.MF = MF
        self.annotator = Annotator(MF, gui_width=1800)    
        self.particles = MF._get_particles()
        self.clicks = []
        self.appearance_model = AppearanceModel(APPEARANCE_MODEL_C_H, 
        APPEARANCE_MODEL_C_S, APPEARANCE_MODEL_C_V, 
        APPEARANCE_MODEL_S_TH, APPEARANCE_MODEL_V_LTH,
        APPEARANCE_MODEL_V_UTH)

        self.motion_model = MotionModel(MOTION_MODEL_SGIMA)



    def  Update(self, detections, frame,original_frame):
        
        if(len(self.tracks) == 0):  
            self.first_frame = frame
            self.InitTracks3(detections, frame,original_frame)
            return

        # initialing motion model and appearance model 
        self.p_motion = np.zeros(shape=(len(self.tracks), len(detections)))
        self.p_color = np.zeros(shape=(len(self.tracks), len(detections)))

        # global state
        dtn_by_trk = {}
        trk_by_dtn = {}
        self.frame_count +=1
        #reformate the detections data
        new_detections = np.zeros((len(detections),2,1))
        for i in range(len(detections)):
            detections[i] =np.array([[detections[i][0]],[detections[i][1]]])
        for n, track in enumerate(self.tracks):
            # get prediction
            predicted_state = track.KF.predict()
            track.prediction = predicted_state



            # looping over detections and connect them to tracks
            for m, detection in enumerate(detections):
                # calc motion model
                p_mo =  self.motion_model.calc(predicted_state, detection)
                self.p_motion[n][m] = p_mo
                
                # if predicted state is probable  
                if self.p_motion[n][m] > P_MOTION_TH :    
                    # calc appearance model 
                    a = self._get_BB_as_img(detection, frame)
                    p_col = self.appearance_model.calc(a, track.track_bb)
                    self.p_color[n][m] = p_col


                    if self.p_motion[n][m]*self.p_color[n][m]> P_COLOR_TH *P_MOTION_TH:
                        # adding to global state
                        if n in dtn_by_trk: dtn_by_trk[n].add(m) 
                        else: dtn_by_trk[n] = {m}
                        
                        # inverse dictionary
                        if m in trk_by_dtn: trk_by_dtn[m].add(n)
                        else: trk_by_dtn[m] = {n}
        
        

        ref_det_by_trk = copy.deepcopy(dtn_by_trk)
        for m in trk_by_dtn:
            # maxium for each model
            max_p_colmo = np.max(self.p_color.T[m] * (self.p_motion.T[m]))
            max_p_mo = np.max(self.p_motion.T[m])
            for n in trk_by_dtn[m]:

                p_colmo_n = self.p_color[n][m] * self.p_motion[n][m]
                p_mo_n = self.p_motion[n][m]
                if p_colmo_n < max_p_colmo :
                    if n in dtn_by_trk and m in dtn_by_trk[n]: 
                        dtn_by_trk[n].remove(m)


        
        # initialization for detection -> track mapping
        assignment = [-1] * len(self.tracks)
        sortedTrack = sorted(self.tracks,key= lambda x:x.skipped_frames)


        

        un_assigned_detects = []
        for i in range(len(detections)):
            un_assigned_detects.append(i)
                
            
        for track in sortedTrack:
            id = track.track_id                
            if id in dtn_by_trk and len(dtn_by_trk[id]) > 0:
                
                # no need to loop
                max_p_colmo = -np.inf
                max_detection = None
                for detection in dtn_by_trk[id]:
                    p_mo = self.p_motion[id][detection]
                    p_col = self.p_color[id][detection]
                    curr_p_colmo = p_mo * p_col
                    if curr_p_colmo > max_p_colmo:
                        max_detection = detection
                        max_p_colmo = curr_p_colmo
                assignment[id] = max_detection
                un_assigned_detects.remove(max_detection)



        # getting tracks without detections
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] == -1):
                self.tracks[i].skipped_frames += 1
                un_assigned_tracks.append(self.tracks[i].track_id)

        
        un_assinged_tracks = []
        for track_id,assign in enumerate(assignment):
            if assign == -1 :
                un_assinged_tracks.append(track_id)

        ref_un_assigned_detects = un_assigned_detects.copy()
        ref_un_assigned_tracks = un_assinged_tracks.copy()

        # Save lost tracks (rasen fel halal)
        for track_id in ref_un_assigned_tracks:
            track = self.tracks[track_id]

            max_detect = -1
            max_p = -1
            if self.tracks[track_id].skipped_frames >= 22:
                for detection_id in ref_un_assigned_detects: 
                    # Place holder
                    detection = detections[detection_id]

                    p_mo = self.p_motion[track_id][detection_id]
                    a = self._get_BB_as_img(detection, frame)
                    p_col = self.appearance_model.calc(a, track.track_bb)

                    p_colmo = p_mo * p_col
                    
                    if p_col > 0.85 and p_mo > 0.005 and p_colmo > max_p:
                        max_p = p_colmo
                        max_detect = detection_id

            if max_detect != -1:
                assignment[track_id] = max_detect
                ref_un_assigned_detects.remove(max_detect)
                ref_un_assigned_tracks.remove(track_id)

                
                        



        
        
        # Pass 2 
        for track_id in ref_un_assigned_tracks:
            max_detect = -1
            max_p = -np.inf
            ## DBG
            for detection_id in ref_un_assigned_detects:
                
                curr_p_mot = self.p_motion[track_id][detection_id]
                if curr_p_mot > 0.45 and curr_p_mot > max_p :
                    max_detect = detection_id
                    max_p = curr_p_mot
            
            if max_detect != -1: #and np.max(self.p_motion.T[max_detect]) < 0.97:
                assignment[track_id] = max_detect
                ref_un_assigned_detects.remove(max_detect) 
                ref_un_assigned_tracks.remove(track_id)
                

        

        # Pass 3
        occuluded_trks = []
        for track_id in ref_un_assigned_tracks:
            if track_id in ref_det_by_trk:
                
                for detection_id in ref_det_by_trk[track_id]:
                    p_mo = self.p_motion[track_id][detection_id]
                    p_col = self.p_color[track_id][detection_id]
                    p_colmo = p_mo * p_col 
                    
            if track_id in ref_det_by_trk:# and self.tracks[track_id].skipped_frames >= 5:
                
                max_p_col = 0
                max_p_mo = 0
                max_p_colmo = 0
                max_p_col_det = -1
                max_p_mo_det = -1
                max_p_colmo_det = -1
                for detection_id in ref_det_by_trk[track_id]:
                    p_mo = self.p_motion[track_id][detection_id]
                    p_col = self.p_color[track_id][detection_id]
                    p_colmo = p_mo * p_col 
                    
                    if p_mo > max_p_mo:
                        max_p_mo = p_mo
                        max_p_mo_det = detection_id
                    if p_colmo > max_p_colmo:
                        max_p_colmo = p_colmo
                        max_p_colmo_det = detection_id
                    if p_col > max_p_col:
                        max_p_col = p_col
                        max_p_col_det = detection_id
                    
                if max_p_mo > 0.95 and self.tracks[track_id].skipped_frames >= 3:

                    last_pos = np.array(self.tracks[track_id].last_measurment)
                    occluded_det = detections[max_p_mo_det] 
                    pred_pos = self.tracks[track_id].prediction
                    new_det = np.round(0.2 * last_pos + 0.8 * occluded_det) ## TODO : @ 0.2 reached f 300
                    assignment[track_id] = len(detections)
                    detections.append(new_det)
                    ref_un_assigned_tracks.remove(track_id)
                    occuluded_trks.append(track_id)
                elif max_p_mo > 0.85 and max_p_col > 0.85 and max_p_col_det == max_p_mo_det and self.tracks[track_id].skipped_frames >= 5:#max_p_colmo > 0.894 and max_p_mo_det == max_p_colmo_det: #  == max_p_col_det:
    
                    last_pos = np.array(self.tracks[track_id].last_measurment)
                    occluded_det = detections[max_p_col_det]
                    pred_pos = track.prediction
                    new_det = np.round(np.mean(np.array([last_pos, occluded_det]), axis= 0))
                    new_det = np.round(0.7 * last_pos + 0.3 * occluded_det)
                    pred_pos = self.tracks[track_id].prediction
                    assignment[track_id] = len(detections)
                    detections.append(new_det)
                    ref_un_assigned_tracks.remove(track_id)

                    occuluded_trks.append(track_id)

    
                
        _dist_tracks = np.zeros((len(self.tracks), len(self.tracks)))

        for i, t1 in enumerate(self.tracks):
            for j in range(i+1, len(self.tracks)):
                t2 = self.tracks[j]
                p1 = t1.trace[-1]
                p2 = t2.trace[-1]
                d = (p1[0][0] - p2[0][0])**2 + (p1[1][0] - p2[1][0])**2
                _dist_tracks[i][j] = np.sqrt(d)
                _dist_tracks[j][i] = np.sqrt(d)
        
        for i in range(len(self.tracks)):
            _dist_tracks[i][i] = np.inf
            

        
        # Update KalmanFilter state, lastResults and tracks trace
        for i in range(len(assignment)):
            if(assignment[i] != -1):
                measurement = detections[assignment[i]]
                coords = (measurement[0][0], measurement[1][0])
                if i in occuluded_trks:
                    particle = self.MF.get_nearest_particle(coords)
                else:
                    particle = self.particles[coords]
                self.tracks[i].prediction = self.tracks[i].KF.correct(measurement)
                self.tracks[i].skipped_frames = 0
                
                _min_dist = np.min(_dist_tracks[i])
                if _min_dist > 70 and i not in occuluded_trks:
                    self.tracks[i].track_bb = self._get_BB_as_img(measurement, frame)
                self.tracks[i].trace.append(self.tracks[i].KF.lastResult)
                if i not in occuluded_trks:
                    self.tracks[i].last_measurment = self.tracks[i].KF.lastResult

            else:
                if self.tracks[i].skipped_frames % 4 == 0 and self.tracks[i].skipped_frames <= 8: 
                    self.tracks[i].trace.append(self.tracks[i].KF.predict())
            if(len(self.tracks[i].trace) > self.max_trace_length): del self.tracks[i].trace[0]
            topview_coords = (self.tracks[i].trace[-1][0][0], self.tracks[i].trace[-1][1][0])
            topview_particle = self.MF.get_nearest_particle(topview_coords)
            if topview_particle:
                self.tracks[i].top_pos = topview_particle.q

            

    # Create tracks if no tracks vector found
    def InitTracks3(self,detections,frame,original_frame):
        
        detections = self.annotator.run(1, original_frame, detections)
        for detection in detections:
    
            particle = self.MF.get_nearest_particle((detection[0],detection[1]))
            x,y = particle.q_img
            top_pos = particle.q
            self.makeTrack([[x],[y]],self.first_frame,0,top_pos)
            
    # Create tracks if no tracks vector found
    def InitTracks2(self,detections,frame,original_frame):
        debug = True
        if debug:
            for click in CLICKS:
                
                particle = self.MF.get_nearest_particle((click[0],click[1]))
                x,y = particle.q_img
                top_pos = particle.q
                self.makeTrack([[x],[y]],self.first_frame,click[2],top_pos)

        if debug:return    
        self.gui_img = original_frame.copy()
        self.gui_img = imutils.resize(self.gui_img, width=GUI_WIDTH)
        self.done = False
        self._write_hint("choose the Home goal keeper")

        cv.namedWindow("GUI")
        cv.setMouseCallback('GUI', self.clickEvent)
        while True:
            cv.imshow('GUI', self.gui_img)
            if self.done:
                cv.waitKey(2000)
                cv.destroyAllWindows()
                break
            cv.waitKey(1)
    def _write_hint(self, msg, color=(0, 0, 0)):
        cv.rectangle(self.gui_img, (10, 2), (300, 20), (255, 255, 255), -1)
        cv.putText(self.gui_img, msg, (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color)  
    def makeTrack(self,point,frame,team,top_pos):
        track_bb = self._get_BB_as_img(point,frame)
        cv.imwrite(f'players/{self.next_trackId}.bmp',track_bb)
        track = Track(point,self.next_trackId,track_bb,team,top_pos)
        self.next_trackId +=1 
        self.tracks.append(track)

    def clickEvent(self, event, x, y, flags=None, params=None):
        # checking for left mouse clicks
        if event != cv.EVENT_LBUTTONDOWN : return 
        self.clicks.append(self._gui2orig((x, y)))
        particle = self.MF.get_nearest_particle((self.clicks[-1][0],self.clicks[-1][1]))
        x,y = particle.q_img
        top_pos = particle.q
        if self.init_state == GuiState.STATE_HOME_GOAL_KEEPER:
            self.makeTrack([[x],[y]],self.first_frame,0,top_pos)
            self.init_state = GuiState.STATE_HOME
            self._write_hint("choose the Home team")
            print(self.clicks)
            self.clicks = []
        
        
        elif self.init_state == GuiState.STATE_HOME:
            self.makeTrack([[x],[y]],self.first_frame,0,top_pos)
            
            if len(self.clicks) < 10:
                curr_click = self.clicks[-1]
                self.gui_img = cv.circle(
                    self.gui_img, curr_click, THICKNESS, RED_COLOR, cv.FILLED)
            else:
                self.init_state = GuiState.STATE_AWAY_GOAL_KEEPER
                self._write_hint("choose the Away team goal keeper")
                print(self.clicks)
                self.clicks = []

        
        elif self.init_state == GuiState.STATE_AWAY_GOAL_KEEPER:

            self.makeTrack([[x],[y]],self.first_frame,1,top_pos)
            self.init_state = GuiState.STATE_AWAY
            self._write_hint("choose the Away team")
            print(self.clicks)
            self.clicks = []
        

        elif self.init_state == GuiState.STATE_AWAY:
            self.makeTrack([[x],[y]],self.first_frame,1,top_pos)
            if len(self.clicks) < 10:
                curr_click = self.clicks[-1]
                self.gui_img = cv.circle(
                    self.gui_img, curr_click, THICKNESS, RED_COLOR, cv.FILLED)
            else:
                self.init_state = GuiState.STATE_REF
                self._write_hint('pick the refree')
                print(self.clicks)
                self.clicks = []
        elif self.init_state == GuiState.STATE_REF:
                self.makeTrack([[x],[y]],self.first_frame,2,top_pos)
                self.done = True
                print(self.clicks)
                self.clicks = []
                self.init_state = GuiState.STATE_END


    def _get_BB_as_img(self, coords, frame):
        coords = (coords[0][0], coords[1][0])
        B = self.particles[coords].B
        tl = B.tl
        br = B.br
        BB_height = br[1] - tl[1]
        BB_width = br[0] - tl[0]

        a = frame[tl[1]:int(tl[1]+BB_height), 
        tl[0]:int(tl[0]+BB_width)]

        return a.copy()

    def _euclidean_dist(self, p1, p2):
        diff = p1 - p2
        dist = np.sqrt(diff[0][0]**2+diff[1][0]**2)
        return dist

    

    def _gui2orig(self, p):
        x = p[0] * self.first_frame.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.first_frame.shape[0] // self.gui_img.shape[0]
        return (x, y)
        
