import numpy as np
import os, sys, time, random, cv2
import matplotlib.pyplot as plt
sift = cv2.xfeatures2d.SIFT_create()

# https://docs.opencv.org/master/d6/d55/tutorial_table_of_content_calib3d.html
img1 = cv2.imread(r'C:\Users\alpha\Desktop\Informatique\TIPE\Images\jpg\test_001_1_60cm.jpg')
img2 = cv2.imread(r'C:\Users\alpha\Desktop\Informatique\TIPE\Images\jpg\test_001_2_60cm.jpg')

def gris(pic):
    return(cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY))

def resize(pic, facteur):
    u, v, _ = pic.shape
    return(cv2.resize(pic, (int(np.floor(u/facteur)), int(np.floor(v/facteur)))))

def convo(pic, kernel):
    res = cv2.filter2D(pic,-1, kernel)
    return(res)

def Canny(pic):
    edges = cv2.Canny(pic, 50, 200, True)
    return(edges)


def Kmeans(pic, k, max_iter, epsilon):
    temps = time.time()
    Z = pic.reshape((-1,3))  #le nb de valeurs finales (les nombres) est cst quelque soit la forme, et -1 signifie que l'ordinateur le calcule lui meme(ex: si 2*3, 6 elt donc reshape((6, -1)) assignera 1 à -1
    Z = np.float32(Z)    #change le type en float
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, epsilon)
    """" Define criteria = ( type, max_iter , epsilon)"""
    ret, label, center = cv2.kmeans(Z , k , None , criteria , 10 , cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((pic.shape))
    print(time.time() - temps)
    return res2

def plot_(pic):
    cv2.namedWindow(nom, cv2.WINDOW_NORMAL)
    cv2.imshow('test', pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#appliquer les k-means sur les x,y pour déterminer des lignes ??

# img = cv2.imread(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\Images\jpg\test.jpg')[:511]
# plot(img, 'dieu est grand')

t = time.time()
img = cv2.imread(r'/home/edern/Documents/TIPE/traitement/mesures/test_image_000030.jpg')

orb = cv2.ORB_create(10000)
kp, des = orb.detectAndCompute(img,None)
img_b = cv2.drawKeypoints(img,kp, img, color=(0,255,0), flags=0)
print(time.time() - t)
plt.imshow(img_b),plt.show()