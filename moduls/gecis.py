import cv2
import numpy as np

def minutiae_at(pixels, i, j, kernel_size): #kesisme sayisi yontemi sirt uclarini ve catallari tespit etmek icin kullanilmakta
    #geçiş sayısı algoritması 3x3 piksel bloklarına bakılacaktir #orta piksel siyahsa (sırt icin)
    if pixels[i][j] == 1: # pixelin boyutu bir ise
        if kernel_size == 3: #3x3 boyut matris oluşturulur
            cells = [(-1, -1), (-1, 0), (-1, 1),        # p1 p2 p3
                   (0, 1),  (1, 1),  (1, 0),            # p8    p4
                  (1, -1), (0, -1), (-1, -1)]           # p7 p6 p5
        else:
            cells = [(-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),                 # p1 p2 p3
                   (-1, 2), (0, 2),  (1, 2),  (2, 2), (2, 1), (2, 0),               # p8    p4
                  (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), (-2, -2)]           # p7 p6 p5
        values = [pixels[i + l][j + k] for k, l in cells] #0'dan 1'e kaç kez gectigi sayilir
        crossings = 0
        for k in range(0, len(values)-1):
            crossings += abs(values[k] - values[k + 1]) #sifirdan bire gitme sayisi
        crossings //= 2
        if crossings == 1: #sinirdaki piksel sırtla bir kez kesisirse
            return "ending" #sırt sonudur
        if crossings == 3: #sinirdaki piksel sırtla uc kez kesisirse
            return "bifurcation" #sirt catalanmasidir
    return "none"

def calculate_minutiaes(im, kernel_size=3): #3x3 matris ile minutiaes noktalarını hesaplayan fonksiyon
    biniry_image = np.zeros_like(im)
    biniry_image[im<10] = 1.0
    biniry_image = biniry_image.astype(np.int8)
    (y, x) = im.shape
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) #resim renkli hale getirilir
    colors = {"ending" : (150, 0, 0), "bifurcation" : (0, 150, 0)} #catallanma nokt. mavi, uc nokt. kırmızı olur 
    for i in range(1, x - kernel_size//2): #her piksel minutia'ını itere ettik
        for j in range(1, y - kernel_size//2):
            minutiae = minutiae_at(biniry_image, j, i, kernel_size) #her piksel icin minutai yineleme
            if minutiae != "none":
                cv2.circle(result, (i,j), radius=2, color=colors[minutiae], thickness=2) #noktalar r=2 olan daireler ile belirtildi 
    return result