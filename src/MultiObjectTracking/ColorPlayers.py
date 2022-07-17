import cv2 as cv
import imutils
from enum import Enum
import numpy as np
from MultiObjectTracking.helper import _write_hint as write_number

TOP_VIEW_WIDTH = 500

COLORS = [(255,255,255),(0,0,255),(0,255,255)]
class GuiState(Enum):
    STATE_CORNERS = 1,
    STATE_GOAL = 2,

class ColorPlayers:
    def __init__(self,tracks_img, canvas, gui_width=800, remove_th=10,
    skipped_frames=0) -> None:
        self.gui_width = gui_width
        self.q_img = []
        self.gui_state = GuiState.STATE_CORNERS
        self.img = None
        self.gui_img = None
        self.frame_id = None
        self.remove_th = remove_th
        self.skipped_frames = skipped_frames
        self.tracks_img = tracks_img
        self.canvas = canvas
        self.hint = ""
        self.info = ''
    def run(self, frame_id, img, top_pos,colors):
        """
        Descritption : this function take the user input to correct the players' teams
         
        Input :
        - frame_id : int
        - img : matrix top field frame
        - colors : int [] current players' teams
        - top_pos : list of tupls (int,int) players' positions from top view
        Output:
        - colors : int [] players' teams after user correction

        """
        self.top_pos = top_pos
        self.img = img
        self.frame_id = frame_id
        self.colors = colors
        self.canvas.set_top_view_callback(self._click_event)
        self._draw_top_pos()

        keypress = None
        while keypress != 13:
            self.canvas.show_canvas(self.tracks_img,top_view = self.gui_img, status=self.hint,info = self.info)
            keypress = cv.waitKey(1)
        return self.colors
    def _write_info(self,msg):
        self.info = msg
    def _draw_top_pos(self):
        temp_img = self.img.copy()
        player_count = 0
        for top_pos_i,team in zip(self.top_pos,self.colors):
            x, y = top_pos_i
            x_offset = 10
            if player_count <10:
                x_offset = 5

            cv.circle(temp_img, (x, y), 10, COLORS[team], -1)
            write_number(temp_img, str(player_count), 
                    np.array([[top_pos_i[0]-x_offset],[top_pos_i[1]+5]]),font = 0.5)

            player_count+=1
        self.gui_img = imutils.resize(temp_img, width=TOP_VIEW_WIDTH)

        self._write_hint("modifying players teams")
        self._write_info('''Press left click to toggle color \n 
Press right click to choose refree \n
Press enter to continue.''')
    
    def _gui2orig(self, p):
        x = p[0] * self.img.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.img.shape[0] // self.gui_img.shape[0]
        return (x, y)

    def _euclidean_distance(self, p1, p2):
        dst = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return dst

    def _write_hint(self, msg):
        self.hint = msg

    def _get_nearest_player(self,click):
        min_dist = 100
        selected_track =  None
        click = np.array([[click[0]],[click[1]]])
        
        for i,track in enumerate(self.top_pos):
            current_dist = self._euclidean_distance(click,track)
            if min_dist > current_dist:
                min_dist = current_dist
                selected_track = i
        return selected_track 

    def _click_event(self, event, x, y, flags=None, params=None):
        x, y = self._gui2orig((x, y))
        if event == cv.EVENT_LBUTTONDOWN :
            nearest_player = self._get_nearest_player((x,y))
            if nearest_player  is not None:
                if self.colors[nearest_player] ==2:
                    self.colors[nearest_player] =1
                self.colors[nearest_player] = 1-self.colors[nearest_player]
            self._draw_top_pos()
        elif event == cv.EVENT_RBUTTONDOWN :
            nearest_player = self._get_nearest_player((x,y))
            if nearest_player  is not None:
                self.colors[nearest_player] = 2
            self._draw_top_pos()