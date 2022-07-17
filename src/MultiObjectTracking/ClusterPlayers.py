from sklearn.cluster import KMeans
import numpy as np
import cv2
import os

class ClustringModule:
    def __init__(self):
        pass
    def getTeamsColors(self):
        """
        Description : returns the initail guess for each player's team
        Output: 
        - colors : list of players' colors
        """
        dominanteColors = []
        imgs = self._load_images_from_folder('./.players')
        for img in imgs:
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            Mask = (img[:, :, 0:3] > [30,30,30]).all(2)
            img = img[Mask]
            try:
                kmeans = KMeans(n_clusters=2, random_state=0).fit(img)
            except:
                return np.zeros(len(imgs),dtype = int)
            dominant_colors = kmeans.cluster_centers_
            indexlist = np.argsort( np.apply_along_axis( np.linalg.norm, 1, dominant_colors))
            dominant_colors = dominant_colors[indexlist]
            if np.linalg.norm(dominant_colors[0]) < np.linalg.norm(dominant_colors[1]):
                dominant_colors[0],dominant_colors[1]  = dominant_colors[1],dominant_colors[0]
            dominanteColors.append(dominant_colors.reshape(6))
            
        dominanteColors =  np.array(dominanteColors)
        try:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(dominanteColors)
        except:
                return np.zeros(len(imgs),dtype = int)
        colors = kmeans.labels_
        return colors
    
    def _load_images_from_folder(self,folder):
        images = []
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder,filename))
            if img is not None:
                images.append(img)
        return images

    