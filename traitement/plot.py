import cv2, time, sys, os


im = cv2.imread(r'C:\Users\alpha\OneDrive\Bureau\Informatique\TIPE\traitement\mesures\test_image_000041.jpg')
print(im.shape)


im = im[128 * b: 128 * (b + 1), 320 * a: 320 * (a + 1)]
print(im.shape)
# im = im.reshape([1280, 128, 3])
# print(im.shape)
# im = im[320:640]
# im = im.reshape([128, 320, 3])

# cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

