from __future__ import absolute_import, division, print_function
import sys
import cv2
import os
import time
import threading
import pygame
from PyQt5.QtGui import QImage
sys.path.append('/home/edern/Documents/TIPE/pyparrot')
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

isAlive = False
# sift = cv2.xfeatures2d.SIFT_create()

class UserVision:
    def __init__(self, vision):
        self.vision = vision

def draw_current_photo():
    return None


#####

#  your code here !

def vol_photo(bebopVision, args):
    bebop = args[0]
    bebop.smart_sleep(2)
    # bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)

    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("help me")
    pygame.event.get()
    userVision = UserVision(bebopVision)
    run = True
    a = 0
    while run:
        pygame.event.get()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]: #z
            run = False
        if keys[pygame.K_s]: #s
            img = userVision.vision.get_latest_valid_picture()[:511]
            filename = r'/home/edern/Documents/TIPE/traitement/mesures/test_image_%06d.jpg' % a
            cv2.imwrite(filename, img)
            a += 1
            print(a)
        pygame.time.delay(150)
    pygame.quit()
    bebop.smart_sleep(3)
    bebop.disconnect()


#####


if __name__ == "__main__":
    global bebop
    bebop = Bebop()
    success = bebop.connect(5)
    if (success):
        bebop.set_video_framerate("24_FPS")
        bebop.set_video_resolutions("rec720_stream720")
        bebop.set_video_stream_mode("high_reliability_low_framerate")
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run = vol_photo, user_args=(bebop, ), user_draw_window_fn=draw_current_photo)
        userVision = UserVision(bebopVision)
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")