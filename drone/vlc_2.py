from __future__ import absolute_import, division, print_function
import sys
import cv2
import os
import time
import threading
import pygame
from PyQt5.QtGui import QImage
sys.path.append('/home/edern/Documents/TIPE')
from pyparrot.Bebop import Bebop
from pyparrot.DroneVisionGUI import DroneVisionGUI
import numpy as np
import concurrent.futures
import glob
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
sys.path.append('/home/edern/Documents/TIPE/monodepth2')
import torch
from torchvision import transforms, datasets
import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


#1280 pixels de large donc un angle de (1.03rads/ (co_en_pixels - (1280/2) ) par rapport a l' axe optique.
#il y a vraiment de quoi disserter sur les K-means. 8 clusters fonctionne bien.
#avantage de prendre peu de points sifts pour appuyer le ML: plus rapide, plus robustes.
#éliminer les points en bordure d'image (cotés). Privilégier le déplacement en avant ?
#Pas nécessaire (?), puisque le réseau travaille sur chaque image indépendament.

isAlive = False
# sift = cv2.xfeatures2d.SIFT_create()

class UserVision:
    def __init__(self, vision):
        self.vision = vision

def draw_current_photo():
    return None

def vol_capteurs(bebopVision, args):
    bebop = args[0]
    bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-5, duration=4)
    dico = []
    t = time.time()
    print(bebop.sensors.sensors_dict)
    time.sleep(1)
    print(bebop.sensors.sensors_dict)
    bebop.smart_sleep(5)
    bebop.disconnect()



if __name__ == "__main__":
    bebop = Bebop()
    success = bebop.connect(5)
    if (success):
        bebop.set_video_framerate("24_FPS")
        bebop.set_video_resolutions("rec720_stream720")
        bebop.set_video_stream_mode("high_reliability_low_framerate")
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run = vol_capteurs, user_args=(bebop, ), user_draw_window_fn=draw_current_photo)
        userVision = UserVision(bebopVision)
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")
