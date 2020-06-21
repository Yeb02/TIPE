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


#1280 pixels de large donc un angle de (1.03rads/ (co_en_pixels - (1280/2) ) par rapport a l' axe optique.
#il y a vraiment de quoi disserter sur les K-means. 8 clusters fonctionne bien.
#avantage de prendre peu de points sifts pour appuyer le ML: plus rapide, plus robustes.
#éliminer les points en bordure d'image (cotés). Privilégier le déplacement en avant ?
#Pas nécessaire (?), puisque le réseau travaille sur chaque image indépendament.
#utiliser le flat_trim

isAlive = False
# sift = cv2.xfeatures2d.SIFT_create()

class UserVision:
    def __init__(self, vision):
        self.vision = vision

def draw_current_photo():
    return None

def vol_final(bebopVision, args):
    global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob
    run_glob = True
    auto_mode_glob = False
    mesures_glob = False
    vitesse_glob = 50  #entre 0 (immobile) et 100.
    tilt_glob = 5   #entre 5 et 30.  30 = rapide, 5 = lent. Mieux vaut 5 pour voler près d'obstacles.
    bebop.set_max_tilt(tilt_glob)

    def controles():
        global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob
        positions = np.zeros([4])
        while run_glob:
            pygame.event.get()
            ctrl = np.zeros([4])   #controles. 0 = haut, 1 = avant, 2 = tourner, 3 = droite, 4 = temps d'éxécution
            keys = pygame.key.get_pressed()

            if mesures_glob:
                dicti = bebop.sensors.sensors_dict
                positions += np.array([dicti['GpsLocationChanged_latitude'],
                dicti['GpsLocationChanged_longitude'], dicti['GpsLocationChanged_altitude'], 1])
                # trouver le sonar.
                # pas besoin d'aussi rapide, et le bebop 2 refresh à 10 Hz
            global vitesse_glob
            v = vitesse_glob
            if keys[pygame.K_a]:
                ctrl[3] = - v
            if keys[pygame.K_z]:
                ctrl[1] = v
            if keys[pygame.K_e]:
                ctrl[3] = v
            if keys[pygame.K_q]:
                ctrl[0] = - v
            if keys[pygame.K_s]:
                ctrl[1] = - v
            if keys[pygame.K_d]:
                ctrl[0] = v
            if keys[pygame.K_w]:
                ctrl[2] = v * np.pi / 100
            if keys[pygame.K_x]:
                ctrl[2] = - v * np.pi / 100
            if keys[pygame.K_SPACE]:
                bebop.safe_land(1)
                bebop.safe_takeoff(1)
                bebop.smart_sleep(2)
                ctrl = np.zeros([4])
            if keys[pygame.K_UP]:
                mesures_glob = not mesures_glob
            if keys[pygame.K_DOWN]:
                run_glob = not run_glob
            if keys[pygame.K_ESCAPE]:
                auto_mode_glob = not auto_mode_glob

            bebop.fly_direct(ctrl[0], ctrl[1], ctrl[2], ctrl[3], .1)
            pygame.time.delay(100)

        return positions

    def auto_mode():    #à faire. Utiliserai le ML pour éviter les obstacles, sur un parcours prédéfini par des injonctions de mouvement ou des cos GPS.
        global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob
        sift = cv2.xfeatures2d.SIFT_create()
        color_pics = []

        while run_glob:
            if auto_mode_glob:
                pass

        time.sleep(.5)

    bebop = args[0]
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("control window")
    pygame.event.get()
    bebop.smart_sleep(1)


    with concurrent.futures.ThreadPoolExecutor() as executor:   #ne sort pas du contexte manager tant que tout les programmes n' ont pas fini.
        thread_1 = executor.submit(controles)
        position = thread_1.result()

    bebop.smart_sleep(3)
    if bebop.safe_land(5):   #n' enleve le controle à l'utilisateur que si le drone a bien atteri.
        pygame.quit()
        bebop.disconnect()



if __name__ == "__main__":
    global bebop
    bebop = Bebop()
    success = bebop.connect(5)
    if (success):
        bebop.set_video_framerate("24_FPS")
        bebop.set_video_resolutions("rec720_stream720")
        bebop.set_video_stream_mode("high_reliability_low_framerate")
        bebopVision = DroneVisionGUI(bebop, is_bebop=True, user_code_to_run = vol_final, user_args=(bebop, ), user_draw_window_fn=draw_current_photo)
        userVision = UserVision(bebopVision)
        bebopVision.open_video()
    else:
        print("Error connecting to bebop.  Retry")
