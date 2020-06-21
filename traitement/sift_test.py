import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import arctan, tan, sin, cos, array #concatenate


t1 = time.time()
img1 = cv2.imread(r'/home/edern/Documents/TIPE/traitement/mesures/test_image_000061.jpg')
img2 = cv2.imread(r'/home/edern/Documents/TIPE/traitement/mesures/test_image_000062.jpg')
# cv2.GaussianBlur(img1,(5,5),0)
# cv2.GaussianBlur(img1,(5,5),0)
# img1 = cv2.undistort(img1, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]
# img2 = cv2.undistort(img2, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]
# img1 = cv2.Canny(img1, 50, 200, True)
# img2 = cv2.Canny(img2, 50, 200, True)


def show(pic, nom):
    cv2.imshow(nom, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



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


def edit(img1, img2):
    pass

def plot(pts1, pts2):
    print(len(pts1), len(pts2))
    t1 = time.time()
    delta = 1              #distance parcourue par le drone en marche avant
    x, y, z = [], [], []
    # pts1 = pts1.reshape(2, l)
    # pts2 = pts2.reshape(2, l)

    # nb = float('%.4f'%(nb)) le nb entre les parentheses pour le réfuire a 4 décimales. Comparer les perfs.
    for p1, p2 in zip(pts1, pts2):
        Ma = np.sqrt((p1[0] - 640)**2 + (p1[1] - 360)**2)
        Mb = np.sqrt((p2[0] - 640)**2 + (p2[1] - 360)**2)

        alpha = Ma / 1280
        beta = Mb / 1280
        tb = tan(beta)
        ta = tan(alpha)
        d = delta * ta / (tb - ta)
        if abs(d) < 10:    #nécessaire pour éviter que les erreurs donnant un point tres lointain rendent le tout inutilisable
            x.append(d * tb * (p2[0] - 640) / Mb)
            z.append(d * tb * (360 - p2[1]) / Mb)
            y.append(d)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x, y, z)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    l = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
    for x1, y1, z1 in zip(x, y, z):
        c = l[np.random.randint(0, 8)]
        ax.scatter(x1, y1, z1, color=c)

    ax.set_xlabel('Z')  # profondeur  (z géométrique)
    ax.set_ylabel('Y')   # latéral ( géométrique)
    ax.set_zlabel('X')   #altitude  ( géométrique)

    print(time.time() - t1)
    plt.show()
    return(x, y, z)



def sift_test(img1, img2):
    # img1 = Kmeans(img1, 3, 20, .5)
    # img2 = Kmeans(img2, 3, 20, .5)
    t = time.time()
    sift = cv2.xfeatures2d.SIFT_create()  #mettre nb keypoints dans les parentheses

    # kp1 = []
    # kp2 = []
    # des1 = []
    # des2 = []
    # for a in range(4):
    #     for b in range(4):
    #         im = img1[128 * b: 128 * (b + 1), 320 * a: 320 * (a + 1)]
    #         u, v = sift.detectAndCompute(im,None)
    #         kp1.append(u)
    #         des1.append(v[0])
    #         im = img2[128 * b: 128 * (b + 1), 320 * a: 320 * (a + 1)]
    #         u, v = sift.detectAndCompute(im,None)
    #         kp2.append(u)
    #         des2.append(v[0])
    # des1 = np.concatenate([des1])
    # des1 = np.concatenate([des2])


    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # desb_1 = []
    # desb_2 = []
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            # desb_1.append(des1[i])
            # desb_2.append(des2[i])

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    print(time.time() - t)

    plt.subplot(121)
    k = []
    l = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
    for elt in pts1:
        c = l[np.random.randint(0, 8)]+ 'o'
        k.append(c)
        plt.plot(elt[0], elt[1], c)
    plt.subplot(122)
    for i, elt in enumerate(pts2):
        c = k[i]
        plt.plot(elt[0], elt[1], c)

    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)
    plt.show()
    return(pts1, pts2)



def sift_plot(img1, img2):          #test mélangeant le plot et le sift pour avoir les couleurs correctes, à faire.

    t = time.time()
    sift = cv2.xfeatures2d.SIFT_create()  #mettre nb keypoints dans les parentheses

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    print(time.time() - t)


    t1 = time.time()
    delta = 1              #distance parcourue par le drone en marche avant
    x, y, z = [], [], []


    # nb = float('%.4f'%(nb)) le nb entre les parentheses pour le réfuire a 4 décimales. Comparer les perfs.

    cmpt = []
    i = 0 #le enumerate dysfonctionne avec le zip dans la boucle suivante

    for p1, p2 in zip(pts1, pts2):

        Ma = np.sqrt((p1[0] - 640)**2 + (p1[1] - 360)**2)
        Mb = np.sqrt((p2[0] - 640)**2 + (p2[1] - 360)**2)

        alpha = Ma / 1280
        beta = Mb / 1280
        tb = tan(beta)
        ta = tan(alpha)
        d = delta * ta / (tb - ta)
        if abs(d) < 10:    #nécessaire pour éviter que les erreurs donnant un point tres lointain rendent le tout inutilisable
            x.append(d * tb * (p2[0] - 640) / Mb)
            z.append(d * tb * (360 - p2[1]) / Mb)
            y.append(d)
            cmpt.append(i)
        i += 1



    k = []
    l = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
    for elt in cmpt:

        c = l[np.random.randint(0, 8)]
        k.append(c)
        c += 'o'

        plt.subplot(121)
        plt.plot(pts1[elt][0], pts1[elt][1], c)

        plt.subplot(122)
        plt.plot(pts2[elt][0], pts2[elt][1], c)

    # plt.subplot(222)
    # for i, elt in enumerate(pts2):
    #     c = k[i]
    #     plt.plot(elt[0], elt[1], c)



    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)




    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    i = 0  #le enumerate dysfonctionne avec le zip dans la boucle suivante
    for x1, y1, z1 in zip(x, y, z):
        c = k[i]
        ax.scatter(x1, y1, z1, color=c)
        i += 1

    ax.set_xlabel('Z')  # profondeur  (z géométrique)
    ax.set_ylabel('Y')   # latéral ( géométrique)
    ax.set_zlabel('X')   #altitude  ( géométrique)

    print(time.time() - t1)
    plt.show()


def edit_depthmap():
    img = cv2.imread(r'/home/edern/Documents/TIPE/traitement/mesures/test_image_000062_disp.jpg')
    n = 5
    im = cv2.resize(im, (int(shape[1]/n), int(shape[0]/n)))
    for a in range(im.shape[0]):
        for b in range(im.shape[1]):


    plt.imshow(img)
    plt.show()





def orb_test(img1, img2):   #inefficace ?
    t = time.time()
    # img1 = Kmeans(img1, 16, 20, .5)
    # img2 = Kmeans(img2, 16, 20, .5)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    img_1 = cv2.drawKeypoints(img1, kp1, img1, color=(0,255,0), flags=0)
    img_2 = cv2.drawKeypoints(img2, kp2, img2, color=(0,255,0), flags=0)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)[:200]

    print(time.time() - t)

    pts1 = [kp1[mat.queryIdx].pt for mat in matches]
    pts2 = [kp2[mat.trainIdx].pt for mat in matches]

    print(time.time() - t)

    t5 = time.time()
    plt.subplot(121)
    k = []
    l = ['b', 'r', 'g', 'y', 'c', 'm', 'k', 'w']
    for elt in pts1:
        c = l[np.random.randint(0, 8)]+ 'o'
        k.append(c)
        plt.plot(elt[0], elt[1], c)
    plt.subplot(122)
    for i, elt in enumerate(pts2):
        c = k[i]
        plt.plot(elt[0], elt[1], c)

    plt.subplot(121)
    plt.imshow(img1)
    plt.subplot(122)
    plt.imshow(img2)

    print(time.time() - t5)

    plt.show()

