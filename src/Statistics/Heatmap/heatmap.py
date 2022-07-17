import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class Heatmap():
    def __init__(self, 
        shade = True,
        n_levels=10,
        cmap = 'magma') -> None:
    
        self.shade = shade
        self.n_levels = n_levels
        self.cmap = cmap

  
    
    def save_heatmaps(self, input_folder, output_folder=""):
        img = plt.imread("h.png")
        h, w, _ = img.shape
        fps = 25

        for track_id in tqdm (range (22), desc="Loading..."):
            if not os.path.exists(f"{input_folder}/{track_id}.csv"):
                continue
            df = pd.read_csv(f"{input_folder}/{track_id}.csv")
            track_df = df[::fps]

            x = track_df.x
            y = track_df.y

            fig, ax = plt.subplots()

            ax.imshow(img)

            if len(track_df.index) > 2:
                sns.kdeplot(
                        x=x,
                        y=y,
                        shade = self.shade,
                        n_levels=self.n_levels,
                        cmap = self.cmap
                )
            else:
                print(f"[WARNING]: not enough data for track #{track_id} to create a heatmap.")

            plt.xlim((0, w))
            plt.ylim((0, h))

            plt.savefig(f"{output_folder}/{track_id}.png")

            plt.close(fig)

def main():
    hm = Heatmap(cmap="icefire")
    
    hm.save_heatmaps("stats", output_folder="heatmaps")

if __name__ == "__main__":
    main()
