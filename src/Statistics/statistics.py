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
    try:
      if not os.path.exists(f"{input_folder}/heatmaps"):
        os.makedirs(f"{input_folder}/heatmaps")
      self.heatmap.save_heatmaps(f"{input_folder}/stats", f"{input_folder}/heatmaps")
      if not os.path.exists(f"{input_folder}/trajectory"):
        os.makedirs(f"{input_folder}/trajectory")
      self.trajectory.save_trajectory(f"{input_folder}/stats", f"{input_folder}/trajectory")

      if not os.path.exists(f"{input_folder}/average_formation"):
        os.makedirs(f"{input_folder}/average_formation")
      self.trajectory.save_distance(f"{input_folder}/stats", f"{input_folder}/average_formation")
    except:
      print('This data is not valid')