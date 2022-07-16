import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pickle 

class Trajectory():
    def __init__(self, MF) -> None:
        self.MF = MF
  
    
    def save_trajectory(self, input_folder, output_folder=""):
        img = plt.imread("h.png")
        h, w, _ = img.shape
        fps = 25
        speed_th = 7

        for track_id in tqdm (range (22), desc="Loading..."):
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

            plt.savefig(f"{output_folder}/{track_id}.png")

            plt.close(fig)

def main():
    filehandler = open(f"/home/mhomran/gp/src/modelField.pkl","rb")
    MF = pickle.load(filehandler)
    filehandler.close()

    tr = Trajectory(MF)
    tr.save_trajectory("stats", output_folder="trajectory")

main()
