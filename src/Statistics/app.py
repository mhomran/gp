import sys
sys.path.append(sys.path[0]+"/../")

from Statistics.statistics import Statistics
import pickle

if __name__=="__main__":
  with open('.model_field.pkl', 'rb') as f:
    MF = pickle.load(f)
    stats = Statistics(MF, f"{sys.path[0]}/../../data/output")
    stats.save_statistics()