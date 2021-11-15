import cv2
import numpy as np
from math import sqrt

def normalize_piksel(x, v0, v, m, m0): #normalizasyon fonksiyonu olusturma
    #x: piksel degeri, v0: istenilen varyans, v: genel resim varyansi m: genel resim ortalamasi, m0: istenilen ortalama
    dev_coeff = sqrt((v0 * ((x - m)**2)) / v) #girilen degerlerin ort. degeri alinir ve normalizasyon icin islem yapılır
    return m0 + dev_coeff if x > m else m0 - dev_coeff #piksel degeri goruntu ort. buyukse istenilen ortalama ile tahmini ort. toplamı, degilse farkı alınır 

def normalize(im, m0, v0):
    m = np.mean(im) #gorselin ortalamasi
    v = np.std(im) ** 2 #gorselin standart sapmasinin karesi
    (y, x) = im.shape #fotografin y ve x boyut bilgileri alinir.
    normilize_image = im.copy() #goruntu normalize image nesnesine aktariliyor
    for i in range(x):
        for j in range(y):
            normilize_image[j, i] = normalize_piksel(im[j, i], v0, v, m, m0)
    #her piksel ayri ayri normalize edilerek normalize image nesnesine atanir
    return normilize_image