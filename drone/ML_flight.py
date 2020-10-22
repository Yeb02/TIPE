from __future__ import absolute_import, division, print_function
import sys
import cv2
import os
import time
import threading
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

import argparse
import pygame
import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
envpath = '/home/edern/.virtualenvs/cv/lib/python3.6/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


cv2.namedWindow('face', cv2.WINDOW_AUTOSIZE)
cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
isAlive = False
# sift = cv2.xfeatures2d.SIFT_create()

class UserVision:
    def __init__(self, vision):
        self.vision = vision

def draw_current_photo():
    return None


######## MONODEPTH
ext='jpg'
model_name='mono_640x192'
#model_name='mono_1024x320'   ### # NOTE: changer pour la taille en dessous si trop lent
no_cuda=False

if torch.cuda.is_available() and not no_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

download_model_if_doesnt_exist(model_name)
model_path = os.path.join("models", model_name)
print("-> Loading model from ", model_path)
encoder_path = os.path.join(model_path, "encoder.pth")
depth_decoder_path = os.path.join(model_path, "depth.pth")

# LOADING PRETRAINED MODEL
print("   Loading pretrained encoder")
encoder = networks.ResnetEncoder(18, False)
loaded_dict_enc = torch.load(encoder_path, map_location=device)

# extract the height and width of image that this model was trained with
feed_height = loaded_dict_enc['height']
feed_width = loaded_dict_enc['width']
filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
encoder.load_state_dict(filtered_dict_enc)
encoder.to(device)
encoder.eval()

print("Loading pretrained decoder")
depth_decoder = networks.DepthDecoder(
    num_ch_enc=encoder.num_ch_enc, scales=range(4))

loaded_dict = torch.load(depth_decoder_path, map_location=device)
depth_decoder.load_state_dict(loaded_dict)

depth_decoder.to(device)
depth_decoder.eval()

print('Monodepth setup.')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

############### CODE BEBOP


def vol_final(bebopVision, args):
    global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob, depth_glob, face_rec_glob
    run_glob = True
    face_rec_glob = False
    depth_glob = False
    auto_mode_glob = False
    mesures_glob = False
    vitesse_glob = 20  #entre 0 (immobile) et 100.
    tilt_glob = 5   #entre 5 et 30.  30 = rapide, 5 = lent. Mieux vaut 5 pour voler près d'obstacles.
    bebop.set_max_tilt(tilt_glob)
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    pygame.display.set_caption("la classe")
    pygame.event.get()

    def depth_display():
        global run_glob, depth_glob
        cptd = 0
        while run_glob:
            if depth_glob:
                print('computing depth')
                path_ = '/home/edern/Documents/TIPE/traitement/mesures/test_image_%03d.jpg' % cptd
                output_directory = os.path.dirname(path_)
                im = userVision.vision.get_latest_valid_picture()
                cv2.imwrite(r'' + path, userVision.vision.get_latest_valid_picture())
                with torch.no_grad():
                    if path_.endswith("_disp.jpg"):
                        continue
                    # Load image and preprocess
                    input_image = pil.open(path_).convert('RGB')
                    original_width, original_height = input_image.size
                    input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                    # PREDICTION
                    input_image = input_image.to(device)
                    features = encoder(input_image)
                    outputs = depth_decoder(features)
                    disp = outputs[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (original_height, original_width), mode="bilinear", align_corners=False)
                    # Saving numpy file
                    output_name = os.path.splitext(os.path.basename(path_))[0]                         #MAP ?
                    name_dest_npy = os.path.join(output_directory, "{}_disp.npy".format(output_name))
                    scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
                    np.save(name_dest_npy, scaled_disp.cpu().numpy())
                    # Saving colormapped depth image
                    disp_resized_np = disp_resized.squeeze().cpu().numpy()
                    vmax = np.percentile(disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)
                    name_dest_im = os.path.join(output_directory, "{}_disp.jpg".format(output_name))
                    im.save(name_dest_im)
                    cv2.imshow(im, 'depth')
                cptd += 1



    def controles():
        print("control given")
        global run_glob, tilt_glob, mesures_glob, bebop, tilt_glob, vitesse_glob, auto_mode_glob, face_rec_glob, depth_glob
        positions = np.zeros([4])
        while run_glob:

            ctrl = np.zeros([4])   #controles. 0 = haut, 1 = avant, 2 = tourner, 3 = droite, 4 = temps d'éxécution
            pygame.event.get()
            keys = pygame.key.get_pressed()

            if face_rec_glob:
                img = userVision.vision.get_latest_valid_picture()
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.imshow('face', img)
                cv2.waitKey(1)
                #cv2.destroyAllWindows()


            if keys[pygame.K_s]:
                filename = r'/home/edern/Documents/TIPE/traitement/mesures/test_image_%06d.jpg' % a
                cv2.imwrite(filename, img)
                print(a)
                a += 1

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
            if keys[pygame.K_f]:
                face_rec_glob = not face_rec_glob
            if keys[pygame.K_SPACE]:
                if bebop.safe_land(2):
                    bebop.smart_sleep(2)
                    ctrl = np.zeros([4])
                else:
                    bebop.safe_takeoff(1)

            if keys[pygame.K_UP]:
                mesures_glob = not mesures_glob
            if keys[pygame.K_DOWN]:
                run_glob = not run_glob
            if keys[pygame.K_ESCAPE]:
                depth_glob = not depth_glob

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
        thread_2 = executor.submit(depth_display)
        position = thread_1.result()


    bebop.smart_sleep(3)
    if bebop.safe_land(5):   #n' enleve le controle à l'utilisateur que si le drone a bien atteri.
        pygame.quit()
        vc.release()
        bebop.disconnect()

############# MAIN

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
