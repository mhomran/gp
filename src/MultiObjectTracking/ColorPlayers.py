import cv2 as cv
import imutils
from enum import Enum
import numpy as np
from MultiObjectTracking.helper import _write_hint as write_number
COLORS = [(255,255,255),(0,0,255),(0,255,255)]
class GuiState(Enum):
    STATE_CORNERS = 1,
    STATE_GOAL = 2,

class ColorPlayers:
    def __init__(self, gui_width=800, remove_th=10,
    skipped_frames=0) -> None:
        self.gui_width = gui_width
        self.q_img = []
        self.gui_state = GuiState.STATE_CORNERS
        self.img = None
        self.gui_img = None
        self.frame_id = None
        self.remove_th = remove_th
        self.skipped_frames = skipped_frames

        cv.namedWindow("GUI-ChooseColors")
        cv.setMouseCallback('GUI-ChooseColors', self._click_event)

    def run(self, frame_id, img, top_pos,colors):
        self.top_pos = top_pos
        self.img = img
        self.frame_id = frame_id
        self.colors = colors
        self._draw_top_pos()

        keypress = None
        while keypress != 27:
            cv.imshow('GUI-ChooseColors', self.gui_img)
            keypress = cv.waitKey(1)
        cv.destroyAllWindows()
        return self.colors

    def _draw_top_pos(self):
        print('update')
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
        self.gui_img = imutils.resize(temp_img, width=self.gui_width)

        msg = str(self.frame_id) + " "
        msg += "you can change any player color "
        self._write_hint(msg)

    
    def _gui2orig(self, p):
        x = p[0] * self.img.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.img.shape[0] // self.gui_img.shape[0]
        return (x, y)

    def _euclidean_distance(self, p1, p2):
        dst = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return dst

    def _write_hint(self, msg, color=(0, 0, 0), pt1=(10, 2), pt2=(600, 20),font = 0.5):
        cv.rectangle(self.gui_img, pt1, pt2, (255, 255, 255), -1)
        cv.putText(self.gui_img, msg, (pt1[0]+5, pt1[1]+13),
                    cv.FONT_HERSHEY_SIMPLEX, font, color)

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
            if nearest_player:
                if self.colors[nearest_player] ==2:
                    self.colors[nearest_player] =1
                self.colors[nearest_player] = 1-self.colors[nearest_player]
            self._draw_top_pos()
        elif event == cv.EVENT_RBUTTONDOWN :
            nearest_player = self._get_nearest_player((x,y))
            if nearest_player:
                self.colors[nearest_player] = 2
            self._draw_top_pos()