# import section 
import numpy as np
from MultiObjectTracking.kalmanfilter import KalmanFilter
import os 
from MultiObjectTracking.AppearanceModel import AppearanceModel
from MultiObjectTracking.MotionModel import MotionModel
from MultiObjectTracking.Annotator import Annotator
from MultiObjectTracking.TrackWriter import TrackWriter
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
    def __init__(self, MF, canvas,base_path = '.'):
        
        self.max_frames_to_skip = MAX_FRAME_SKIP
        self.max_trace_length = MAX_TRACE_LEN
        self.next_trackId = START_TRACK_ID
        self.tracks = []
        self.p_motion = None
        self.p_color = None
        self.base_path = base_path
        self.init_state = GuiState.STATE_HOME_GOAL_KEEPER
        self.frame_count = 0
        self.MF = MF
        self.annotator = Annotator(MF, canvas, gui_width=1700)    
        
        self.csv_files_initialized = False
        
        self.particles = MF._get_particles()
        self.clicks = []
        self.appearance_model = AppearanceModel(APPEARANCE_MODEL_C_H, 
        APPEARANCE_MODEL_C_S, APPEARANCE_MODEL_C_V, 
        APPEARANCE_MODEL_S_TH, APPEARANCE_MODEL_V_LTH,
        APPEARANCE_MODEL_V_UTH)

        self.motion_model = MotionModel(MOTION_MODEL_SGIMA)
        self.canvas = canvas


    def  Update(self, detections, frame,original_frame):
        """
        Description: update the internal tracker state and tracks locations 

        Input:
            - detections : array of detections 
            - frame : The current masked frame
            - original_frame : The current original frame
        """        
        if(len(self.tracks) == 0):  
            self.first_frame = frame
            self._init_tracks(detections,original_frame)
            return

        # initialing motion model and appearance model 
        self.p_motion = np.zeros(shape=(len(self.tracks), len(detections)))
        self.p_color = np.zeros(shape=(len(self.tracks), len(detections)))

        self.frame_count +=1
        #reformate the detections data
        for i in range(len(detections)):
            detections[i] =np.array([[detections[i][0]],[detections[i][1]]])
        # global state
        dtn_by_trk,trk_by_dtn,ref_det_by_trk = self._calc_global_state(detections,frame)
        # initialization for detection -> track mapping
        assignment = [-1] * len(self.tracks)
        
        # getting detections without tracks
        un_assigned_detects = []
        for i in range(len(detections)):
            un_assigned_detects.append(i)

        self._initial_guess(dtn_by_trk,assignment,un_assigned_detects)
        # getting tracks without detections
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if (assignment[i] == -1):
                self.tracks[i].skipped_frames += 1
                un_assigned_tracks.append(self.tracks[i].track_id)
        
        ref_un_assigned_detects = un_assigned_detects.copy()
        ref_un_assigned_tracks = un_assigned_tracks.copy()

        # pass1
        self._pass1(detections,frame,assignment,ref_un_assigned_tracks,ref_un_assigned_detects)
        # Pass 2 
        self._pass2(assignment,ref_un_assigned_tracks,ref_un_assigned_detects)
        # Pass 3
        occuluded_trks = self._pass3(detections,assignment,ref_un_assigned_tracks,ref_det_by_trk)
        # Update KalmanFilter state, lastResults and tracks trace
        self._update_kalman(assignment,detections,occuluded_trks,frame)
        # Write tracks trace to file
        self._write_tracks_to_disk(detections,assignment)
    
    def _initial_guess(self,dtn_by_trk,assignment,un_assigned_detects):
        sortedTrack = sorted(self.tracks,key= lambda x:x.skipped_frames)
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
                if max_detection in un_assigned_detects:
                    un_assigned_detects.remove(max_detection)
    
    def _pass1(self,detections,frame,assignment,ref_un_assigned_tracks,ref_un_assigned_detects):
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
    def _pass2(self,assignment,ref_un_assigned_tracks,ref_un_assigned_detects):
        for track_id in ref_un_assigned_tracks:
            max_detect = -1
            max_p = -np.inf
            ## DBG
            for detection_id in ref_un_assigned_detects:
                
                curr_p_mot = self.p_motion[track_id][detection_id]
                if curr_p_mot > 0.45 and curr_p_mot > max_p :
                    max_detect = detection_id
                    max_p = curr_p_mot
            
            if max_detect != -1: 
                assignment[track_id] = max_detect
                ref_un_assigned_detects.remove(max_detect) 
                ref_un_assigned_tracks.remove(track_id)
                
    def _pass3(self,detections,assignment,ref_un_assigned_tracks,ref_det_by_trk):
        occuluded_trks = []
        for track_id in ref_un_assigned_tracks:
            if track_id in ref_det_by_trk:
                
                for detection_id in ref_det_by_trk[track_id]:
                    p_mo = self.p_motion[track_id][detection_id]
                    p_col = self.p_color[track_id][detection_id]
                    p_colmo = p_mo * p_col 
                    
            if track_id in ref_det_by_trk:
                
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
                    
                    new_det = np.round(np.mean(np.array([last_pos, occluded_det]), axis= 0))
                    new_det = np.round(0.7 * last_pos + 0.3 * occluded_det)
                    assignment[track_id] = len(detections)
                    detections.append(new_det)
                    ref_un_assigned_tracks.remove(track_id)

                    occuluded_trks.append(track_id)
        return occuluded_trks

    def _calc_global_state(self,detections,frame):
        dtn_by_trk = {}
        trk_by_dtn = {}
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
        return dtn_by_trk,trk_by_dtn,ref_det_by_trk
        
    def _update_kalman(self,assignment,detections,occuluded_trks,frame):
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
    
    def _initialize_csv_files(self):
        exists = os.path.exists(self.base_path +"/stats")
        if not exists:
            
            os.makedirs(self.base_path +  "/stats")

        for id in range(len(self.tracks)):
            print(id)
            if self.tracks[id].team == 2:
                continue
            TrackWriter.initialize_file(self.base_path +  "/stats" + f'/{id}.csv')

    def _write_tracks_to_disk(self,detections,assignment):

        if not self.csv_files_initialized:
            self._initialize_csv_files()
            self.csv_files_initialized = True
        for track,detection_id in zip(self.tracks,assignment):
            if track.team == 2:
                continue
            if detection_id ==-1:
                detection = [[-1],[-1]]
            else:
                detection = detections[detection_id]

            TrackWriter.write(f'{self.base_path}/stats/{track.track_id}.csv',self.frame_count,track,detection)




    # Create tracks if no tracks vector found
    def _init_tracks(self,detections,original_frame):
        exists = os.path.exists("./.players")
        if not exists:
            os.makedirs("./.players")

        detections = self.annotator.run(1, original_frame, detections)
        for detection in detections:
    
            particle = self.MF.get_nearest_particle((detection[0],detection[1]))
            x,y = particle.q_img
            top_pos = particle.q
            self._make_track([[x],[y]],self.first_frame,0,top_pos)
        


    def _write_hint(self, msg, color=(0, 0, 0)):
        cv.rectangle(self.gui_img, (10, 2), (300, 20), (255, 255, 255), -1)
        cv.putText(self.gui_img, msg, (15, 15),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, color) 

    def _make_track(self,point,frame,team,top_pos):
        track_bb = self._get_BB_as_img(point,frame)
        cv.imwrite(f'./.players/{self.next_trackId}.bmp',track_bb)
        track = Track(point,self.next_trackId,track_bb,team,top_pos)
        self.next_trackId +=1 
        self.tracks.append(track)

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
        
