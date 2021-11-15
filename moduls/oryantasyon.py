import cv2
import numpy as np
import math

def calculate_angles(im, W, smoth=False):
    #im: resim , W: sirtin genisligi (int.)
    j1 = lambda x, y: 2 * x * y #ileride kullanilmak icin gerceklestirilen islemler
    j2 = lambda x, y: x ** 2 - y ** 2
    j3 = lambda x, y: x ** 2 + y ** 2
    (y, x) = im.shape #fotografin y ve x boyut bilgileri alinir.
    sobelOperator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] #sobel filtresi icin matris olusturma
    ySobel = np.array(sobelOperator).astype(np.int) #sobeloperator dizinin int turune donusturulmus kopyasi
    xSobel = np.transpose(ySobel).astype(np.int)  #ysobel dizisi transpoze ederek (eksenleri tersine çevirme) ve degistirme; değiştirilen dizi dondurulur
    result = [[] for i in range(1, y, W)]
    Gx_ = cv2.filter2D(im/125,-1, ySobel)*125 #resme rastgele doğrusal filtre uygulandıktan sonra Gx_ değişkenine atanir
    Gy_ = cv2.filter2D(im/125,-1, xSobel)*125 #resme rastgele doğrusal filtre uygulandıktan sonra Gy_ değişkenine atanir

    for j in range(1, y, W):
        for i in range(1, x, W):
            nominator = 0 #nominator değişkenine 0 atama
            denominator = 0 #denominator değişkenine 0 atama
            for l in range(j, min(j + W, y - 1)):
                for k in range(i, min(i + W , x - 1)):
                    Gx = round(Gx_[l, k])  #l, k da yatay gradyan oluşturduktan sonra Gx in ondalıklı kısmı döndürüldü 
                    Gy = round(Gy_[l, k])  #l, k da düşey gradyan oluşturduktan sonra Gy in ondalıklı kısmı döndürüldü 
                    nominator += j1(Gx, Gy) #nomınator ile j1 in Gx. Ve Gy. Noktraları toplanir
                    denominator += j2(Gx, Gy)# denomınator ile j2 in Gx. Ve Gy. Noktraları toplanir

            if nominator or denominator:
                angle = (math.pi + math.atan2(nominator, denominator)) / 2 #pay ve paydanin arctanjantini alinip pi sayısı ile toplanir
                oryantasyon = np.pi/2 + math.atan2(nominator,denominator)/2 #pi/2 ile pay ve paydanin arctan toplanir
                result[int((j-1) // W)].append(angle) #sonuca angle eklenerek result içine atanir
            else:
                result[int((j-1) // W)].append(0) #sonuca sifir eklenerek result içine atanir
    result = np.array(result) #sonucun diziye çevrilmesi

    if smoth: #eğer düzleşti ise 
        result = smooth_angles(result) # sonuçtaki açıları duzlestir
    return result

def gauss(x, y): #gauss fonksiyonu
    ssigma = 1.0 # sigma degeri 1.0 olarak atandi
    return (1 / (2 * math.pi * ssigma)) * math.exp(-(x * x + y * y) / (2 * ssigma)) # gaus işlemleri

def kernel_from_function(size, f): #cekirdek fonksiyonu
    kernel = [[] for i in range(0, size)] #kernel icine sifirdan girilen boyut degerine kadar her i sayisi icin sayı atanir
    for i in range(0, size):
        for j in range(0, size):
            kernel[i].append(f(i - size / 2, j - size / 2)) #kernel dizisinde her i ye yukarıdaki matematiksel dizi eklenir
    return kernel

def smooth_angles(angles):
    angles = np.array(angles) #angeles dizi haline getirildi
    cos_angles = np.cos(angles.copy()*2) #dizinın kopyasinin iki ile çarpımı cosinüs alir
    sin_angles = np.sin(angles.copy()*2) #dizinın kopyasinin iki ile çarpımı sinus alir
    kernel = np.array(kernel_from_function(5, gauss)) #olusturalan kernel fonksiyonuna çekirdek ve gauss fonksiyonu girilir
    cos_angles = cv2.filter2D(cos_angles/125,-1, kernel)*125 #görüntüye rastgele dogrusal filtre uygulanir ve cos_angles değişkenine atanir
    sin_angles = cv2.filter2D(sin_angles/125,-1, kernel)*125 #görüntüye rastgele dogrusal filtre uygulanir ve sin_angles değişkenine atanir
    smooth_angles = np.arctan2(sin_angles, cos_angles)/2 #iki degerin arctan degeri ile acisi bulunur
    return smooth_angles

def get_line_ends(i, j, W, tang): # biten çizgileri bulmak için olusturulan fonksiyon 
    if -1 <= tang and tang <= 1: #eğer tan a değeri -1 ile 1 arasında ise
        begin = (i, int((-W/2) * tang + j + W/2)) #baslangic olarak i ve yandaki degeri al
        end = (i + W, int((W/2) * tang + j + W/2)) # baslangictan sonra bitis için i+w, ve denklemin sonucunu al
    else:
        begin = (int(i + W/2 + W/(2 * tang)), j + W//2) #koordinat baslangic i,j değeri icin yukaridaki degerleri al
        end = (int(i + W/2 - W/(2 * tang)), j - W//2) #koordinat bitis i,j değeri icin yukaridaki degerleri al
    return (begin, end)

def visualize_angles(im, mask, angles, W): #acilari gorsellestirmek icin olusturulan fonksiyon
    (y, x) = im.shape #fotografin y ve x boyut bilgileri alinir.
    result = cv2.cvtColor(np.zeros(im.shape, np.uint8), cv2.COLOR_GRAY2RGB) #resimi gri renkten bgr renklendirmeli resme çevir
    mask_threshold = (W-1)**2 # threshold (w) degerinden bir çıkarıp iki ile çarparak maske oluşturulur
    for i in range(1, x, W):
        for j in range(1, y, W):
            radian = np.sum(mask[j - 1:j + W, i-1:i+W]) #x ve y düzlemindeki noktalar toplanarak resimdeki yaricap uzunluğundaki yayi gören merkez aciya eşit aci
            if radian > mask_threshold: #radian maske esik degerinden büyükse
                tang = math.tan(angles[(j - 1) // W][(i - 1) // W]) #tan degeri alinir
                (begin, end) = get_line_ends(i, j, W, tang) #biten cizgiler bulunup begin,end değerine atanir
                cv2.line(result, begin, end, color=150) # edinilen sonuçlar, başlangıçtan bitis pikseline kadar gri bir çizgi ile cizdirilir
    cv2.resize(result, im.shape, result) #yeniden boyutlandirma
    return result