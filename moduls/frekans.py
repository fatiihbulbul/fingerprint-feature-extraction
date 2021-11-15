import cv2
import numpy as np
import math
import scipy.ndimage

def frequest(im, orientim, kernel_size, minWaveLength, maxWaveLength):
    #bir parmak izi resminin kucuk bir blogu iCindeki sırt frekansini tahmin etme işlevi
    #bir tepe frekansi bulunamazsa veya min ve maks ile belirlenen sinirlar dahilinde bulunamazsa, dalga boyu frekansı sıfır olarak ayarlanır
    rows, cols = np.shape(im) #fotografin y ve x boyut bilgileri alinir.
    cosorient = np.cos(2*orientim) #blok içindeki ortalama yönelimi bulduktan sonra açıyı yeniden yapılandırmadan once
    sinorient = np.sin(2*orientim) #iki katina çıkan açıların sinüslerinin ve kosinüslerinin ortalamasını alarak yapılır
    block_orient = math.atan2(sinorient,cosorient)/2
    rotim = scipy.ndimage.rotate(im,block_orient/np.pi*180 + 90,axes=(1,0),reshape = False,order = 3,mode = 'nearest') #goruntu blogu sırtlar dikey olacak sekilde dondurulur
    cropsze = int(np.fix(rows/np.sqrt(2))) #döndürülen görüntünün geçersiz bölge içermemesi için goruntu kirpilir
    offset = int(np.fix((rows-cropsze)/2))
    rotim = rotim[offset:offset+cropsze][:,offset:offset+cropsze]
    ridge_sum = np.sum(rotim, axis = 0) #sirtlardaki gri değerlerin yansımasını elde etmek için sutunlar toplanir
    dilation = scipy.ndimage.grey_dilation(ridge_sum, kernel_size, structure=np.ones(kernel_size))
    ridge_noise = np.abs(dilation - ridge_sum); peak_thresh = 2;
    maxpts = (ridge_noise < peak_thresh) & (ridge_sum > np.mean(ridge_sum))
    maxind = np.where(maxpts)
    _, no_of_peaks = np.shape(maxind)
    
    if(no_of_peaks<2):
        freq_block = np.zeros(im.shape)
    else:
        waveLength = (maxind[0][-1] - maxind[0][0])/(no_of_peaks - 1) #birinci ve son zirveler arasındaki mesafe (Tepe sayısı-1) 'e bolunerek
        if waveLength>=minWaveLength and waveLength<=maxWaveLength: #sirtlarin uzamsal frekansi belirlenir
            freq_block = 1/np.double(waveLength) * np.ones(im.shape)
        else: #hic bir tepe algılanmazsa veya dalga boyu izin verilen sınırların dışındaysa
            freq_block = np.zeros(im.shape) #frekans görüntüsü 0 olarak ayarlanır
    return(freq_block)

def ridge_freq(im, mask, orient, block_size, kernel_size, minWaveLength, maxWaveLength): # Parmak izi görüntüsü boyunca parmak izi sırt frekansını tahmin etmek icin fonk.
    rows,cols = im.shape
    freq = np.zeros((rows,cols)) #sifir ile yeni bir dizi olarak döner 
    for row in range(0, rows - block_size, block_size): 
        for col in range(0, cols - block_size, block_size): 
            image_block = im[row:row + block_size][:, col:col + block_size] #her bir sütun ve satır icin
            angle_block = orient[row // block_size][col // block_size] #resim  ve aci bloklar olusturulur
            if angle_block:
                freq[row:row + block_size][:, col:col + block_size] = frequest(image_block, angle_block, kernel_size, minWaveLength, maxWaveLength)

    freq = freq*mask #frekansı mask ile carpilir
    freq_1d = np.reshape(freq,(1,rows*cols)) #frekans dizisine verilerini değiştirmeden yeniden sekillendirme 
    ind = np.where(freq_1d>0) #freq_1d in sifirdan buyuk oldugu yerler ind içine yazilir
    ind = np.array(ind) #ind diziye çevrilir
    ind = ind[1,:] #ind arrayın birden küçük değerleri kesilir
    non_zero_elems_in_freq = freq_1d[0][ind] 
    medianfreq = np.median(non_zero_elems_in_freq) * mask #medyan degeri
    return medianfreq