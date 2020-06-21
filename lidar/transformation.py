import sys
import numpy as np
import cv2

vidcap = cv2.VideoCapture(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\lidar\video-1579523430.mp4')
success,image = vidcap.read()
print(success)
count = 0
while success:
  cv2.imwrite(r"C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\lidar\images\frame%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1