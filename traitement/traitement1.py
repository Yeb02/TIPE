import matplotlib.image as img
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import time
import os
os.chdir(r"C:\Users\alpha\Desktop\Informatique\TIPE\Images")

plt.axis("off")

# R1 = np.array([[ np.random.randint (100) for i in range(10) ] for j in range(10) ])
# R2 = np.array([[sqrt(i**2 + j**2) for i in range(100)] for j in range(100)])
# R2[50, 50] = 200
# plt.imshow (R1 , cmap = 'jet')                        #couleurs en jet




pic = img.imread('oiseau1.png')  
a = pic.shape


# red = np.zeros(a)
# Rgray = np.zeros(a)
# Vgray = np.zeros(a)
# Bgray = np.zeros(a)
Truegray = np.zeros(a)
for i in range(a[0]):
    for j in range(a[1]):  
#         red[i, j, 0] = pic[i, j, 0]                 #couleurs RVB (juste rouge ici)
#         Rgray[i, j] = [pic[i, j, 0]] * 3            #gris avec les 3 couleurs
#         Vgray[i, j] = [pic[i, j, 1]] * 3
#         Bgray[i, j] = [pic[i, j, 2]] * 3
        Truegray[i, j] = [(pic[i, j, 0] + pic[i, j, 1] + pic[i, j, 2])/3] * 4  #gris moyen
        
plt.subplot(131)
plt.imshow(pic)
# plt.subplot(122)
# plt.imshow(red)                     
                             # Les deux ne veulent pas s' afficher simultanément....
# plt.subplot(221)
# plt.imshow(Rgray)
# plt.subplot(222)
# plt.imshow(Vgray)
# plt.subplot(223)
# plt.imshow(Bgray)
# plt.subplot(224)
plt.subplot(132)
plt.imshow(Truegray)


def ring(filtre, i, j):
    k, s = 0, 0
    for a in range(-1, 2):       #à modifier pour un filtre plus grand, mais pour l' instant on optimise
        for b in range(-1, 2):
            k = k + filtre[a + 1, b + 1] * Truegray[i + a, j + b, 0]
            s = s + filtre[a + 1, b + 1] 
    return(k / s)
        
brd = 1
co = 1
mater = np.zeros((a[0], a[1], a[2] - 1))            #attention -1
filtre = np.array([[12, 0, 0], [0, 1, 0], [1, 0, 0]])               #produits de convolution
for i in range(brd, a[0] - brd):
    for j in range(brd, a[1] - brd):
        # mater[i, j] = [ring(filtre, i, j)] * 3
        
        q = (Truegray[i - brd, j, 0] - Truegray[i + brd, j, 0]) ** 2
        r = (Truegray[i, j - brd, 0] - Truegray[i, j + brd, 0]) ** 2
        if q > 0.03 or r > 0.03:                            #filtre bordures  grises  (double 0.03)
            mater[i, j] = [0] * 3
        else:
            mater[i, j] = [1] * 3
 
        # k = 0
        # for co in range(3):
        #     a1 = (pic[i - brd, j, co] - pic[i + brd, j, co]) ** 2
        #     a2 = (pic[i, j - brd, co] - pic[i, j + brd, co]) ** 2
        #     a3 = (pic[i - brd, j + brd, co] - pic[i + brd, j - brd, co]) ** 2 
        #     a4 = (pic[i - brd, j - brd, co] - pic[i + brd, j + brd, co]) ** 2
        #     k += (a1 + a2 + a3 +a4)/(4 * 3)
        # mater[i, j] = [k] * 3
        
                                           
        
plt.subplot(133)
plt.imshow(mater)


# plt.axis("off")
plt.show ()

