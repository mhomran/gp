from Statistics.Heatmap.heatmap import Heatmap
from Statistics.Trajectory.trajectory import Trajectory
import os
class Statistics:
  def __init__(self, MF, input_folder='.') -> None:
    self.heatmap = Heatmap(cmap="icefire")
    self.trajectory = Trajectory(MF)

    self.input_folder = input_folder

  def save_statistics(self):
    input_folder = self.input_folder
    if not os.path.exists(f"{input_folder}/heatmaps"):
      os.makedirs(f"{input_folder}/heatmaps")
    self.heatmap.save_heatmaps(input_folder, f"{input_folder}/heatmaps")
    
    if not os.path.exists(f"{input_folder}/trajectory"):
      os.makedirs(f"{input_folder}/trajectory")
    self.trajectory.save_trajectory(input_folder, f"{input_folder}/trajectory")
