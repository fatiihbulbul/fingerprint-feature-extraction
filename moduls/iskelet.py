import cv2
import numpy as np
from moduls.gecis import calculate_minutiaes
#from skimage.morphology import skeletonize as skelt
#from skimage.morphology import thin

def skeletonize(image_input): #2d dizi uint8 (0-65535 arasında tam sayılar)
    #iskeletlestirme, ikili nesneleri bir piksel boyutunda temsillere indirger. Görüntünün ard arda gecişlerini yapar
    #gecislere, ilgili nesnenin bağlantısını kesmemeleri sartiyle sinir piksellerini belirler ve kaldırır
    image = np.zeros_like(image_input) #giris resmine belirli bir diziyle aynı şekildeki bir sıfır dizisi döndürülür
    image[image_input == 0] = 1.0
    output = np.zeros_like(image_input)
    skeleton = skelt(image) # iskelet yapısı cikti
    output[skeleton] = 255
    cv2.bitwise_not(output, output) #dizinin tüm bitlerini tersine çevirip output içine atar
    return output

def thinning_morph(image, kernel): #morfolojik islemlerle görüntüyü incelten fonksiyon
    #image: 2d dizi uint8, kernel: 3x3 matrisli 2d dizi
    thining_image = np.zeros_like(image)
    img = image.copy() 

    while 1:
        erosion = cv2.erode(img, kernel, iterations = 1) #resimi inceltildi (asindirma)
        dilatate = cv2.dilate(erosion, kernel, iterations = 1) #resim genişletildi
        subs_img = np.subtract(img, dilatate) #bağımsız değişkenler öğe olarak çıkartıldı
        cv2.bitwise_or(thining_image, subs_img, thining_image) #dizinin ve bir skalerin her oge icin bit bazli ayrismasi hesaplanir
        img = erosion.copy()
        done = (np.sum(img) == 0) 
        if done:
          break

    down = np.zeros_like(thining_image)
    down[1:-1, :] = thining_image[0:-2, ] #piksel asagı kaydirilir 
    down_mask = np.subtract(down, thining_image) #ve bir piksel ofsetini karsilastirilir
    down_mask[0:-2, :] = down_mask[1:-1, ]
    cv2.imshow('down', down_mask)
    left = np.zeros_like(thining_image)
    left[:, 1:-1] = thining_image[:, 0:-2] #piksel saga kaydirilir 
    left_mask = np.subtract(left, thining_image) #ve bir piksel ofsetini karsilastirilir
    left_mask[:, 0:-2] = left_mask[:, 1:-1]
    cv2.imshow('left', left_mask)
    cv2.bitwise_or(down_mask, down_mask, thining_image) # sol ve asağı maskeyi birlestir
    output = np.zeros_like(thining_image)
    output[thining_image < 250] = 255
    return output