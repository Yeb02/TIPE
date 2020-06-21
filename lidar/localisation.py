import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# sys.path.append(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\lidar\images')



xdef = []
ydef = []
col = []
i = 15
j = 93


for a in range(i, j):
    im = cv2.imread(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\lidar\images\frame%d.jpg' % (a*10))
    c, xf, yf = 0, 0, 0
    for x in range(368):
        for y in range(640):
            pix = im[x, y]
            if pix[0] > 230 and pix[1] > 210 and pix[2] > 240:
                xf += x
                yf += y
                c += 1
    if c != 0:
        xf /= c
        yf /= c
    xdef.append(int(xf))
    ydef.append(int(yf))
    u = a / 94
    print(a, u)
    plt.plot(xdef[-1], ydef[-1], 'ro')

c, xf, yf = 0, 0, 0
im = cv2.imread(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\lidar\images\frame200.jpg')
for x in range(368):
    for y in range(640):
        pix = im[x, y]
        if pix[0] > 230 and pix[1] > 210 and pix[2] > 240:
            xf += x
            yf += y
            c += 1
if c != 0:
    xf /= c
    yf /= c
plt.subplot(111)
plt.plot(xf, yf, 'ro')
plt.matshow(im)

# im = cv2.imread(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\lidar\images\frame200.jpg')
# plt.matshow(im)
plt.show()

