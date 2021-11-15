import cv2 as cv
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from moduls.normalizasyon import normalize
from moduls.segmentasyon import create_segmented_and_variance_images
from moduls import oryantasyon
from moduls.frekans import ridge_freq
from moduls.gabor_filtresi import gabor_filter
from moduls.iskelet import skeletonize
from moduls.gecis import calculate_minutiaes
from moduls.acisal import calculate_singularities

def fingerprint_feature_extraction(input_img): #islemler sirasiyla normalizasyon, oryantasyon, frekans, maskeleme, filtreleme
    block_size = 16
    normalized_img = normalize(input_img.copy(), float(100), float(100)) #normalizasyon işlemi ile resimdeki gürültü ve basınc farklılıkları yok edilir
    (segmented_img, normim, mask) = create_segmented_and_variance_images(normalized_img, block_size, 0.2) #ROI (İlgi alanın belirlenmesi) ve normalizasyon
    angles = oryantasyon.calculate_angles(normalized_img, W=block_size, smoth=False) #alan oryantasyonu acilar hesaplanir
    orientation_img = oryantasyon.visualize_angles(segmented_img, mask, angles, W=block_size) #alan oryantasyonu acilar hesaplanir
    freq = ridge_freq(normim, mask, angles, block_size, kernel_size=5, minWaveLength=5, maxWaveLength=15) #genel sirt sikliginin bulunmasi
    gabor_img = gabor_filter(normim, angles, freq) # gabor filtresi ile filtreleme islemi
    thin_image = skeletonize(gabor_img) # Inceltme ve iskelet çıkarma islemi
    minutias = calculate_minutiaes(thin_image) # minutia noktlarının cıkarılması    
    singularities_img = calculate_singularities(thin_image, angles, 1, block_size, mask) # eşsiz noktaların bulunması yani farklılık olusturan noktalar
    #yapılan islemlerin sırasıyla gösterilmesi, asagida
    output_imgs = [input_img, normalized_img, segmented_img, orientation_img, gabor_img, thin_image, minutias, singularities_img]
    for i in range(len(output_imgs)):
        if len(output_imgs[i].shape) == 2:
            output_imgs[i] = cv.cvtColor(output_imgs[i], cv.COLOR_GRAY2RGB)
    results = np.concatenate([np.concatenate(output_imgs[:4], 1), np.concatenate(output_imgs[4:], 1)]).astype(np.uint8)
    return results

if __name__ == '__main__': #islenecek resmin cagirilmasi
    img_dir = './girdi/*' #girdi
    output_dir = './cikti/' #ciktilar
    def open_images(directory):
        images_paths = glob(directory)
        return np.array([cv.imread(img_path,0) for img_path in images_paths]) #resmin okunmasi
    images = open_images(img_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(tqdm(images)):
        results = fingerprint_feature_extraction(img)
        cv.imwrite(output_dir+str(i)+'.png', results) #ciktilarin yazdirilmasi
        # cv.imshow('image pipeline', results); cv.waitKeyEx()