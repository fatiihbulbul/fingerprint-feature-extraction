import cv2
import numpy as np
import scipy

def gabor_filter(im, orient, freq, kx=0.65, ky=0.65): #Gabor filtresi, kenar algılama için kullanılan doğrusal bir filtredir
    #Gauss ile modüle edilmis, belirli frekans ve yönelimdeki sinüzoidal düzlem 
    angleInc = 3
    im = np.double(im)  #im double türüne cevrilir 
    rows, cols = im.shape #resmin ordinat degerleri aktarilir
    return_img = np.zeros((rows,cols)) #resim 0 ile dondurulur
    freq_1d = freq.flatten() #incelenmesi gereken frekans sayısını azaltmak için frekans dizisi en yakın 0,01'e yuvarlanir
    frequency_ind = np.array(np.where(freq_1d>0)) #dizide sifirdan buyuk olan frekanslar diziye atanir
    non_zero_elems_in_freq = freq_1d[frequency_ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq*100)))/100
    unfreq = np.unique(non_zero_elems_in_freq)
    sigma_x = 1/unfreq*kx #farklı frekanslar karsilik gelen filtreler olusturulur 
    sigma_y = 1/unfreq*ky #ve 'angleInc' artislariyla yonelimler belirlenir
    block_size = np.round(3*np.max([sigma_x,sigma_y]))
    array = np.linspace(-block_size,block_size,(2*block_size + 1))
    x, y = np.meshgrid(array, array) #koordinat vektorleri, koordinat matrislerini dondurulur 
    reffilter = np.exp(-(((np.power(x,2))/(sigma_x*sigma_x) + (np.power(y,2))/(sigma_y*sigma_y)))) * np.cos(2*np.pi*unfreq[0]*x) #gabor filtre denklemi
    filt_rows, filt_cols = reffilter.shape
    gabor_filter = np.array(np.zeros((180//angleInc, filt_rows, filt_cols)))
    
    for degree in range(0,180//angleInc): #filtrenin döndürülmüş versiyonlarını olusturma
        rot_filt = scipy.ndimage.rotate(reffilter,-(degree*angleInc + 90),reshape = False)
        gabor_filter[degree] = rot_filt
    maxorientindex = np.round(180/angleInc) #oryantasyon matris değerlerini radyan turunden tura(derece/angleInc) karşılık gelen indeks değerine dönüştür
    orientindex = np.round(orient/np.pi*180/angleInc)
    for i in range(0,rows//16):
        for j in range(0,cols//16):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    block_size = int(block_size)
    valid_row, valid_col = np.where(freq>0)
    finalind = \
        np.where((valid_row>block_size) & (valid_row<rows - block_size) & (valid_col>block_size) & (valid_col<cols - block_size))
        #resim sınırından maksimum boyuttan büyük matris noktalarının endekslerini bulunur, ustte
    for k in range(0, np.shape(finalind)[1]):
        r = valid_row[finalind[0][k]]; c = valid_col[finalind[0][k]]
        img_block = im[r-block_size:r+block_size + 1][:,c-block_size:c+block_size + 1]
        return_img[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r//16][c//16]) - 1])
    gabor_img = 255 - np.array((return_img < 0)*255).astype(np.uint8)
    return gabor_img