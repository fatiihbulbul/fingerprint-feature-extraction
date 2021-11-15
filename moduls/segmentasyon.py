import cv2
import numpy as np

def normalise(img):
    return (img - np.mean(img))/(np.std(img)) #resimden resim ort. cikartilarak standart sapmaya bolme islemi

def create_segmented_and_variance_images(im, w, threshold=.2): #esik degeri iki olan fonks.
    #im: resim, w: blok boyutu, threshold: esik deger
    #İlgili alanı tanımlamak icin calisir. Bunun icin her blogun standart sapmasini hesaplar ve ROI'ye esler. Resmin yogunluk degerlerini normallesitir.
    # Bu sayede sırt bolgeleri 0 ortalama ve birim standart sapmaya sahip olur. 
    (y, x) = im.shape #fotografin y ve x boyut bilgileri alinir.
    threshold = np.std(im)*threshold #resmin standart sapmasi ile esik degeri carpiliyor
    image_variance = np.zeros(im.shape) # Sifirlarla doldurularak, verilen şekil ve türde yeni bir dizi dondurulur
    segmented_image = im.copy() #goruntu segmented image nesnesine aktariliyor
    mask = np.ones_like(im) #bir diziyi aynı şekle ve türe sahip bir dizi döndürdük ve mask değişkenine attık

    for i in range(0, x, w):
        for j in range(0, y, w):
            box = [i, j, min(i + w, x), min(j + w, y)] #box degiskeni icine i,j icin sirasiyla  i+w,x’den en dusuk ve j+w,y ‘den en dusuk sayi atanir
            block_stddev = np.std(im[box[1]:box[3], box[0]:box[2]]) #box degiskeninin 1. Ve 3. araligi ile 0. Ve 2. araliginin standart sapması alinir 
            image_variance[box[1]:box[3], box[0]:box[2]] = block_stddev #alinan standart sapma, yeniden 1.ve 3. araliga ve 0.ve 2. araliga atanir

    mask[image_variance < threshold] = 0 #esik degerinin uygulanmasi
    # açık / kapalı morfolojik filtreli pürüzsüz maske
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(w*2, w*2)) #morpholoji elemanlarından elipsin blok boyutunun karesi alinir
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #open morfolojisini olusturarak kernel içindeki eleman ile mask içine atanmasi
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #close morfolojisini olusturarak mask içine atanmasi

    #bolumlenmis goruntunun normallestirilmesi
    segmented_image *= mask #segmented_image ve mask'in carpimi
    im = normalise(im) #resmin normalizesi
    mean_val = np.mean(im[mask==0]) #0. mod alinarak resmin ortalamasi elde edildi
    std_val = np.std(im[mask==0]) #0. mod alinarak resmin standart sapmasi elde edildi
    norm_img = (im - mean_val)/(std_val) #resim ortalama degerden cikartildiktan sonra standart sapmaya bolunmesi
    return segmented_image, norm_img, mask