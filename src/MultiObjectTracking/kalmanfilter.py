import cv2
import numpy as np


class KalmanFilter:
    ACC_VAR = 100
    DT = 1/25
    def __init__(self, x, y) -> None:
        dt = self.DT
        self.kf = cv2.KalmanFilter(4, 2)
        
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                        [0, 1, 0, 0]], np.float32)

        self.kf.transitionMatrix = np.array([[1, 0, dt, 0], 
                                        [0, 1, 0, dt], 
                                        [0, 0, 1, 0], 
                                        [0, 0, 0, 1]], np.float32)

        # variable speed motion model
        self.kf.processNoiseCov = np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                                        [0, (dt**4)/4, 0, (dt**3)/2],
                                        [(dt**3)/2, 0, (dt**2), 0],
                                        [0, (dt**3)/2, 0, (dt**2)]], np.float32)
        self.kf.processNoiseCov *= self.ACC_VAR

        inital_state = np.array([
                    [np.float32(x[0])],
                    [np.float32(y[0])],
                    [np.float32(10)],
                    [np.float32(10)]
                    ])
                    
        self.set_initial_state(inital_state)

        self.lastResult = [x, y]

    def correct(self, measurement=None):
        ''' This function estimates the position of the object'''
        if measurement is None: 
            measurement = self.lastResult

        coordX, coordY = measurement
        coordX = coordX[0]
        coordY = coordY[0]

        measurement = np.array([[np.float32(coordX)], [np.float32(coordY)]])

        corrected = self.kf.correct(measurement)
        
        corrected = [[int(corrected[0])], [int(corrected[1])]]
        self.lastResult = corrected

        return corrected
    
    def predict(self):
        predicted = self.kf.predict()
        predicted = [[int(predicted[0])], [int(predicted[1])]]
        self.lastResult = predicted

        return predicted

    def set_initial_state(self, x):
        self.kf.statePre = x
        self.kf.statePost = x
