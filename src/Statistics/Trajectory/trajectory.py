import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm
import numpy as np
import pickle 
import os
import cv2 as cv
from MultiObjectTracking.helper import _write_hint as write_number

COLORS = [(255,255,255),(0,0,255),(0,255,255)]

class Trajectory():
    def __init__(self, MF) -> None:
        self.MF = MF
  
    
    def save_trajectory(self, input_folder, output_folder=""):
        img = plt.imread("h.png")
        h, w, _ = img.shape
        fps = 25
        speed_th = 7

        for track_id in tqdm (range (22), desc="Loading..."):
            if not os.path.exists(f"{input_folder}/{track_id}.csv"):
                continue
            df = pd.read_csv(f"{input_folder}/{track_id}.csv")
            track_df = df[::fps]

            x = track_df.x
            y = track_df.y

            x_vel = x.diff().values[1:]
            y_vel = y.diff().values[1:]
            y_vel, x_vel = self.MF.convert_px2m((y_vel, x_vel))
            speed = np.sqrt(np.power(x_vel, 2)+np.power(y_vel, 2))
            sprint_mask = speed>speed_th
            x_start = x[:-1][sprint_mask]
            x_end = x[1:][sprint_mask]
            y_start = y[:-1][sprint_mask]
            y_end = y[1:][sprint_mask]
            dx = x_end.values - x_start.values
            dy = y_end.values - y_start.values

            fig, ax = plt.subplots()

            ax.imshow(img)

            sprints = zip(x_start.values, y_start.values, dx, dy)
            for sprint in sprints:
                x_st, y_st, dx, dy = sprint
                plt.arrow(x_st, y_st, dx, dy, head_width=20, head_length=10)

            plt.xlim((0, w))
            plt.ylim((0, h))

            plt.gca().invert_yaxis()

            plt.savefig(f"{output_folder}/{track_id}.png")

            plt.close(fig)

    def save_distance(self, input_folder, output_folder=""):
        img = cv.imread("h.png")

        with open(f"{output_folder}/distance.csv", 'w') as csvfile: 
            writer = csv.writer(csvfile)

            for track_id in tqdm (range (22), desc="Loading..."):
                if not os.path.exists(f"{input_folder}/{track_id}.csv"):
                    continue
                df = pd.read_csv(f"{input_folder}/{track_id}.csv")

                y, x = df.y.values, df.x.values
                avg_formation = int(x.mean()), int(y.mean())
                y, x = self.MF.convert_px2m((y, x))

                x_st = x[:-1]
                y_st = y[:-1]
                x_ed = x[1:]
                y_ed = y[1:]

                dst = np.sqrt(np.power(x_st-x_ed,2)+np.power(y_st-y_ed,2))
                total_dst = dst.sum()
                writer.writerow([str(track_id), total_dst]) 

                x, y = avg_formation
                x_offset = 10
                if track_id <10:
                    x_offset = 5

                cv.circle(img, (x, y), 10, COLORS[df.team.values[0]], -1)
                write_number(img, str(track_id), 
                        np.array([[x-x_offset],[y+5]]),font = 0.5)


        cv.imwrite(f"{output_folder}/avg_formation.png", img)

def main():
    filehandler = open(f"/home/mhomran/gp/src/modelField.pkl","rb")
    MF = pickle.load(filehandler)
    filehandler.close()

    tr = Trajectory(MF)
    tr.save_trajectory("stats", output_folder="trajectory")

if __name__ == "__main__":
    main()
