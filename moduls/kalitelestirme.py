import cv2
from matplotlib import pyplot as plt

def show_img_thresholds(img):
    ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_OTSU)
    ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)
    titles = ['Original Image','BINARY_INV','OTSU','BINARY|OTSU','BINARY','TOZERO_INV']
    images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
    for i in range(6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return thresh2

img = cv2.imread('girdi/fingerprint.jpg',0)
threshold_degerleri= show_img_thresholds(img)
plt.imshow(threshold_degerleri)
plt.show()