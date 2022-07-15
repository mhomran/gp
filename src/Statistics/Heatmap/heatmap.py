import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class Heatmap():
    def __init__(self, 
        shade = True,
        n_levels=10,
        cmap = 'magma') -> None:
    
        self.shade = shade
        self.n_levels = n_levels
        self.cmap = cmap

  
    
    def save_heatmaps(self, fname, output_folder=""):
        df = pd.read_csv(fname)
        img = plt.imread("h.png")
        h, w, _ = img.shape
        fps = 25

        for track_id in tqdm (range (22), desc="Loading..."):
            track_df = df[df.track_id==track_id][::fps]

            x = track_df.x
            y = track_df.y

            fig, ax = plt.subplots()

            ax.imshow(img)

            sns.kdeplot(
                    x=x,
                    y=y,
                    shade = self.shade,
                    n_levels=self.n_levels,
                    cmap = self.cmap
            )

            plt.xlim((0, w))
            plt.ylim((0, h))

            plt.savefig(f"{output_folder}/{track_id}.png")

            plt.close(fig)

def main():
    hm = Heatmap(cmap="icefire")
    
    hm.save_heatmaps("tracks_with_q.csv", output_folder="heatmaps")


main()
