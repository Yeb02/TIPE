import os, sys, time, threading
sys.path.append('C:\ProgramData\Anaconda3\Lib\site-packages')
import numpy as np
import cv2, ffmpeg, os
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision

sys.path.append(r'C:\ProgramData\Anaconda3\Lib\site-packages\ffmpeg')
sys.path.append(r'C:\Users\alpha\Desktop\Informatique\TIPE\ffmpeg-4.2')
sys.path.append(r'C:\Users\alpha\Desktop\Informatique\TIPE\ffmpeg\ffmpeg-4.2-win64-static\bin')

isAlive = False

def Canny(pic):
    edges = cv2.Canny(pic, 50, 200, True)
    return(edges)

class UserVision:
    def __init__(self, vision):
        self.index = 0
        self.vision = vision
        
    def Canny(pic):
        edges = cv2.Canny(pic, 50, 200, True)
        return(edges)

    def save_pictures(self, args):
        #print("saving picture")
        img = self.vision.get_latest_valid_picture()
        if (img is not None):
            img = Canny(img)
            filename = "test_image_%06d.png" % self.index
            cv2.imwrite(filename, img)
            self.index +=1


# make my bebop object
bebop = Bebop()

# connect to the bebop
success = bebop.connect(5)

if (success):
    # start up the video
    bebop.set_video_framerate("24_FPS")
    bebop.set_video_resolutions("rec720_stream720")
    bebopVision = DroneVision(bebop, is_bebop=True)
    bebopVision.cleanup_old_images = True
    userVision = UserVision(bebopVision)
    bebopVision.set_user_callback_function(userVision.save_pictures, user_callback_args=None)
    success = bebopVision.open_video()

    # if (success):
    #     print("Vision successfully started!")
    #     print("Fly me around by hand!")
    #     bebop.smart_sleep(1)
    #     print("Moving the camera using velocity")
    #     bebop.pan_tilt_camera_velocity(pan_velocity=0, tilt_velocity=-2, duration=4)
    #     bebop.smart_sleep(1)
    #     print("Finishing demo and stopping vision")
    #     bebopVision.close_video()

    # disconnect nicely so we don't need a reboot
    bebopVision.close_video() # à enlever
    bebop.disconnect()
else:
    print("Error connecting to bebop.  Retry")