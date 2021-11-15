import cv2
import numpy as np
import math
from moduls import oryantasyon

def poincare_index_at(i, j, angles, tolerance): #fonksiyon once maske olusturulur
    cells = [(-1, -1), (-1, 0), (-1, 1),         # p1 p2 p3
            (0, 1),  (1, 1),  (1, 0),            # p8    p4
            (1, -1), (0, -1), (-1, -1)]          # p7 p6 p5
    angles_around_index = [math.degrees(angles[i - k][j - l]) for k, l in cells] # açıların dereceleri hesaplanir
    index = 0
    for k in range(0, 8):
        difference = angles_around_index[k] - angles_around_index[k + 1] #farkliliklar hesaplanir
        if difference > 90: #doksan dereceden buyukse
            difference -= 180 #yüz seksenden cikar 
        elif difference < -90: #eksi doksandan kucukse
            difference += 180 #yüz seksenden topla 
        index += difference
    if 180 - tolerance <= index <= 180 + tolerance:
        return "loop" #360 derece ise loop olur
    if -180 - tolerance <= index <= -180 + tolerance:
        return "delta" #eksi 180 derece ise delta olur
    if 360 - tolerance <= index <= 360 + tolerance:
        return "whorl" #artı 180 ise whorl olur
    return "none"

def calculate_singularities(im, angles, tolerance, W, mask): #singularityleri hesaplamayan fonksiyon
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) #resim renkli hale getirildi
    colors = {"loop" : (0, 0, 255), "delta" : (0, 128, 255), "whorl": (255, 153, 255)} #loop,delta,whorl renk atamasi yapilir
    # DELTA: RED, LOOP:ORAGNE, whorl:INK
    for i in range(3, len(angles) - 2): #Y
        for j in range(3, len(angles[i]) - 2): #x
            mask_slice = mask[(i-2)*W:(i+3)*W, (j-2)*W:(j+3)*W] #maskenin disindaki herhangi bir essizlik maskelenir
            mask_flag = np.sum(mask_slice) # toplami alinir
            if mask_flag == (W*5)**2:
                singularity = poincare_index_at(i, j, angles, tolerance)
                if singularity != "none": #herhangı bir singularity varsa
                    cv2.rectangle(result, ((j+0)*W, (i+0)*W), ((j+1)*W, (i+1)*W), colors[singularity], 3) #dikdortgen ciz
    return result