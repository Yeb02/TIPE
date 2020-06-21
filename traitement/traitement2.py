import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os, sys, time, random

pic1 = plt.imread(r'C:\Users\alpha\Desktop\Informatique\TIPE\Images\png\test_001_1.png')
pic2 = plt.imread(r'C:\Users\alpha\Desktop\Informatique\TIPE\Images\png\test_001_2.png')



def print_rgb(pic):
    pic_R = np . zeros ([256 , 256 , 3])
    for i in range(256):
        for j in range(256):
            pic_R [i , j , 0] = pic [i , j , 0]
    pic_G = np . zeros ([256 , 256 , 3])
    for i in range (256) :
        for j in range (256) :
            pic_G [i , j , 1] = pic [i , j , 1]
    pic_B = np . zeros ([256 , 256 , 3])
    for i in range (256) :
        for j in range (256) :
            pic_B [i , j , 2] = pic [i , j , 2]
    plt.subplot(2, 2, 1)
    plt.imshow(pic)
    plt.axis("off")
    plt.subplot(2, 2, 2)
    plt.imshow(pic_R)
    plt.axis("off")
    plt.subplot(2, 2, 3)
    plt.imshow(pic_G)
    plt.axis("off")
    plt.subplot(2, 2, 4)
    plt.imshow(pic_B)
    plt.axis("off")
    plt.show()
   
   
   

def convolution(pic , mat, triple) :
    if triple:
        n , p , _ = pic.shape
        chgt = np.zeros([ n -2 , p -2 , 3])
        for i in range (1 , n -1) :
            for j in range (1 , p -1) :
                s = 0
                for u in range (3) :
                    for v in range (3) :
                        s += mat[u , v] * pic[i -1 + u , j -1 + v ]
                for k in range (3) :
                    # chgt [i -1 , j -1 , k ] = s [ k ]
                    if s[k] < 0:
                        chgt [i -1 , j -1 , k ] = 0
                    elif s[k] > 1:
                        chgt [i -1 , j -1 , k ] = 1
                    else :
                        chgt [i -1 , j -1 , k ] = s [ k ]
    else:
        n , p = pic.shape
        chgt = np.zeros([ n -2 , p -2])
        for i in range (1 , n -1) :
            for j in range (1 , p -1) :
                s = 0
                for u in range (3) :
                    for v in range (3) :
                        s += mat[u , v] * pic[i -1 + u , j -1 + v ]
                if s < 0:
                    chgt [i -1 , j -1] = 0
                elif s > 1:
                    chgt [i -1 , j -1] = 1
                else :
                    chgt [i -1 , j -1 ] = s
    return chgt
    # plt.subplot(1, 2, 1)
    # plt.imshow(chgt)
    # plt.subplot(1, 2, 2)
    # plt.imshow(pic)
    # plt.show()


def nv_gris(pic, triple):
    u, v, _ = pic.shape
    if triple:
        gris1 = np . zeros ([u, v, 3])
        # gris2 = np . zeros ([u, v, 3])
        for a in range(u):
            for b in range(v):
                gris1[a, b] = [(pic[a, b][0] + pic[a, b][1] + pic[a, b][2])/3] * 3      # version standard
                # gris2[a, b] = [(pic[a, b][0] * .299 + pic[a, b][1] * .587 + pic[a, b][2] * .114)/3] * 3    # meilleur ?
    else:
        gris1 = np . zeros ([u, v])
        # gris2 = np . zeros ([u, v])
        for a in range(u):
            for b in range(v):
                gris1[a, b] = (pic[a, b][0] + pic[a, b][1] + pic[a, b][2])/3
                # gris2[a, b] = (pic[a, b][0] * .299 + pic[a, b][1] * .587 + pic[a, b][2] * .114)/3 
    return gris1  # ou gris2.       Module pillow pour faire du jpeg, voir mpeg.


def contours( image , seuil) :
    n , p , _ = image . shape
    gris1 = nv_gris( image, True )
    Nlle1 = np . zeros ([ n , p , 3])
    for i in range (1 , n -1) :
        for j in range (1 , p -1) :
            x = gris1 [i , j +1 , 0] - gris1 [i , j -1 , 0]
            y = gris1 [ i +1 , j , 0] - gris1 [i - 1, j , 0]
            norme = np . sqrt ( x ** 2 + y ** 2)
            if norme < seuil :
                Nlle1 [i , j ] = [1 , 1 , 1]             #Python n'ajuste pas automatiquement : [0, 1]
    return ( Nlle1 )                                   #pour les floats, [0, 255] int.   Prends la valeur maximale si depassee. (idem min)
    # plt.subplot(1, 2, 1)
    # plt.imshow(Nlle1)
    # plt.subplot(1, 2, 2)
    # plt.imshow(image)
    # plt.show()


def sobel(pic):
    t0 = time.time()
    M1 = 1/9 * np.ones([3,3])   #flou basique.
    M2 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) * 1/16   # flou gaussien
    M3x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    M3y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    u, v, _ = pic.shape
    edges = np.zeros([u - 4, v - 4, 3])
    g = nv_gris(pic, False)
    g = convolution(g, M2, False)   #réduire le bruit
    x = convolution(g, M3x, False)  #étrange.
    y = convolution(g, M3y, False)
    # return x, y
    for a in range(u - 4):
        for b in range(v - 4):                                  # ↓ sqrt(2)
            edges[a, b] = 3 * [np.sqrt(x[a, b]**2 + y[a, b]**2)/1.41]  
    return(edges, t0)

def exsobel():
    p1, t01 = sobel(pic1)            #Posons que les aires des éléments visibles changent peu. (bords exclus)
    plt.subplot(121)
    plt.imshow(p1)
    print(time.time() - t01)
    p2, t02 = sobel(pic2) 
    plt.subplot(122)
    plt.imshow(p2)
    print(time.time() - t02)
    plt.show()
    
def resizing(pic, taillex, tailley):
    u, v, w = pic.shape
    reshaped = np.zeros([taillex, tailley, w])
    x = u/taillex
    y = v/tailley
    for a in range(taillex):
        for b in range(tailley):
            s = np.zeros([w])
            for c in range(int(np.floor(a*x)), int(np.floor((a+1)*x))):
                for d in range(int(np.floor(b*y)), int(np.floor((b+1)*y))):
                    s += pic[c, d]
            reshaped[a, b] = s / ((int(np.floor((a+1)*x)) - int(np.floor(a*x) - 1)) * (int(np.floor((b+1)*y)) - int(np.floor(b*y)) - 1))
    return(reshaped)
    

    
    
    
    