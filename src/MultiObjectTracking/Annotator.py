
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
    def __init__(self, mf, canvas, gui_width=1200, remove_th=10,
    skipped_frames=0) -> None:
        self.mf = mf
        self.gui_width = gui_width
        self.q_img = []
        self.gui_state = GuiState.STATE_CORNERS
        self.img = None
        self.gui_img = None
        self.frame_id = None
        self.remove_th = remove_th
        self.skipped_frames = skipped_frames
        self.canvas = canvas
        self.hint = ""
        
    def run(self, frame_id, img, q_img):
        self.q_img = q_img
        self.img = img
        self.frame_id = frame_id
        self.canvas.set_callback(self._click_event)

        self._draw_q_img()

        keypress = None
        while keypress != 27:
            self.canvas.show_canvas(self.gui_img, status=self.hint)
            keypress = cv.waitKey(1)
        return self.q_img

    def _draw_q_img(self):
        temp_img = self.img.copy()

        for q_img_i in self.q_img:
            x, y = q_img_i
            cv.circle(temp_img, (x, y), 10, (0,0,255), -1)

        self.gui_img = imutils.resize(temp_img, width=self.gui_width)
        msg = ""
        msg += "You can add or remove a detection.\n"
        msg += "Press esc to continue.\n"
        msg += "Number of players: " + str(len(self.q_img)) + "\n"

        self._write_hint(msg)

    def _gui2orig(self, p):
        x = p[0] * self.img.shape[1] // self.gui_img.shape[1]
        y = p[1] * self.img.shape[0] // self.gui_img.shape[0]
        return (x, y)

    def _euclidean_distance(self, p1, p2):
        dst = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        return dst

    def _write_hint(self, msg):
        self.hint = msg

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
    base_path = "/home/mhomran/Downloads/ground_truth/progression"
    IMG = ImageClass(f'{base_path}/original.avi')
    BGIMG = cv.imread(f'{base_path}/../Common/background.png')

    skipped_frames = 0
    repeated_frames = 3
    q_img = None

    MF = {}
    with open(f'{base_path}/../Common/modelField.pkl', 'rb') as f:
        MF = pickle.load(f)

    PD = PlayerDetection(MF, IMG, BGIMG)
    annotator = Annotator(MF, gui_width=1800, skipped_frames=skipped_frames)

    if not os.path.exists(f"{base_path}/q_img_gt"):
        os.makedirs(f"{base_path}/q_img_gt")

    while True:

        _, frame, frameId = IMG.readFrame()

        if frame is None:
            break
        
        if (frameId-1) % repeated_frames == 0:
            # Player Detection
            fgMask = PD.subBG(frame, frameId)

            PD.preProcessing(fgMask)
            PD.loopOnBB()

            _, q_img = PD.getOutputPD()

            # Annotate
            q_img = annotator.run(frameId, frame, q_img)

        # Save
        if frameId > skipped_frames:
            TagWriter.write(f"{base_path}/q_img_gt/{frameId}.csv", q_img)
        

if __name__=="__main__":
    main()