import os
import cv2
import pandas as pd
import numpy as np
from skimage import feature
from skimage.feature import graycomatrix, graycoprops

def calculate_features(image):
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    asm = graycoprops(glcm, 'ASM')
    
    return np.mean(contrast), np.mean(dissimilarity), np.mean(homogeneity), np.mean(energy), np.mean(correlation), np.mean(asm)

metadata = pd.read_csv('dataset/metadata.csv') # path to your metadata file

dataset_folders = ['train', 'test', 'validation']

new_dataset = []

for folder in dataset_folders:
    image_folder_path = os.path.join('dataset\\images\\', folder) # update with your actual path
    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        image_id = image_name.split('.')[0] # assuming image_name format is "IMAGE_0000244.jpg"
        
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # the texture features are calculated on grayscale images
        
        features = calculate_features(image)
        
        metadata_row = metadata.loc[metadata['image_id'] == image_id].values.tolist()[0]
        new_row = metadata_row + list(features)
        
        new_dataset.append(new_row)

        print(len(new_dataset))

new_dataset_df = pd.DataFrame(new_dataset, columns=metadata.columns.tolist() + ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm'])
new_dataset_df.to_csv('new_dataset.csv', index=False)