import cv2, time, sys, os
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE')

def integrateur():
    vitesse = [np.array([0, 0, 0])]
    position = np.zeros([3])
    while run_glob:
        if mesures_glob_accel:
            vitesse.append(vitesse[-1] + refresh_glob * acceleration_glob)
            position += vitesse[-1] * refresh_glob
        time.sleep(refresh_glob)
    return position