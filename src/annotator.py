from PlayerDetection.PlayerDetection import PlayerDetection
from PlayerDetection.ImageClass import ImageClass
from PlayerDetection.TagWriter import TagWriter
import pickle, os
import cv2 as cv
import imutils
from enum import Enum
import numpy as np

class GuiState(Enum):
    STATE_CORNERS = 1,
    STATE_GOAL = 2,

class Annotator:
    def __init__(self, mf, gui_width=1200, remove_th=10) -> None:
        self.mf = mf
        self.gui_width = gui_width
        self.q_img = []
        self.gui_state = GuiState.STATE_CORNERS
        self.img = None
        self.gui_img = None
        self.frame_id = None
        self.remove_th = remove_th

        cv.namedWindow("GUI")
        cv.setMouseCallback('GUI', self._click_event)

    def run(self, frame_id, img, q_img):
        self.q_img = q_img
        self.img = img
        self.frame_id = frame_id

        self._draw_q_img()

        keypress = None
        while keypress != 27:
            cv.imshow('GUI', self.gui_img)
            keypress = cv.waitKey(1)

        return self.q_img

    def _draw_q_img(self):
        temp_img = self.img.copy()

        for q_img_i in self.q_img:
            x, y = q_img_i
            cv.circle(temp_img, (x, y), 3, (0,0,255), -1)

        self.gui_img = imutils.resize(temp_img, width=self.gui_width)

        self._write_hint(str(self.frame_id))

    def _gui2orig(self, p):
        x = p[0] * self.img.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.img.shape[0] // self.gui_img.shape[0]
        return (x, y)

    def _euclidean_distance(self, p1, p2):
        dst = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return dst

    def _write_hint(self, msg, color=(0, 0, 0)):
        cv.rectangle(self.gui_img, (10, 2), (300, 20), (255, 255, 255), -1)
        cv.putText(self.gui_img, msg, (15, 15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, color)

    def _click_event(self, event, x, y, flags=None, params=None):
        x, y = self._gui2orig((x, y))

        if event == cv.EVENT_LBUTTONDOWN:
            particle = self.mf.get_nearest_particle((x, y))
            if particle is not None:
                self.q_img.append(particle.q_img)
                self._draw_q_img()
        elif event == cv.EVENT_RBUTTONDOWN:
            q_img_temp = []
            for q_img_i in self.q_img:
                dst = self._euclidean_distance(q_img_i, (x, y))
                if dst > self.remove_th:
                    q_img_temp.append(q_img_i)

            self.q_img = q_img_temp
            self._draw_q_img()

def main():
    IMG = ImageClass('two_mins.avi')
    BGIMG = cv.imread('background.png')
    BGIMG = imutils.resize(BGIMG, 1200)

    MF = {}
    with open('MF.pkl', 'rb') as f:
        MF = pickle.load(f)

    PD = PlayerDetection(MF, IMG, BGIMG)
    annotator = Annotator(MF, gui_width=1800)

    if not os.path.exists("q_img"):
        os.makedirs("q_img")

    while True:

        _, frame, frameId = IMG.readFrame()

        if frame is None:
            break

        # Player Detection
        fgMask = PD.subBG(frame, frameId)

        PD.preProcessing(fgMask)
        PD.loopOnBB()

        _, q_img = PD.getOutputPD()

        # Annotate
        q_img = annotator.run(frameId, frame, q_img)

        # Save
        TagWriter.write(f"q_img/{frameId}.csv", q_img)
        

if __name__=="__main__":
    main()