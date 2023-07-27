from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import pandas as pd
import numpy as np
import os
import cv2

def calculate_features(image):
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    return [np.mean(graycoprops(glcm, prop)) for prop in props]

# Loading metadata and initializing new datasets
metadata = pd.read_csv('dataset/metadata.csv')
metadata['age'].fillna(metadata['age'].mean(), inplace = True)

dataset_folders = ['train', 'test', 'validation']
datasets = defaultdict(list)

# Iterating over images
for folder in dataset_folders:
    image_folder_path = os.path.join('dataset\\images\\', folder)
    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        image_id = image_name.split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = calculate_features(image)
        metadata_row = metadata.loc[metadata['image_id'] == image_id].values.tolist()[0]
        datasets[folder].append(metadata_row + features)

        if len(datasets[folder]) % 1000 == 0:
            print(f"Extracted features for {len(datasets[folder])} images in {folder}.")
            break

# Processing each of the datasets separately
for folder in dataset_folders:
    new_dataset_df = pd.DataFrame(datasets[folder], columns=metadata.columns.tolist() + ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm'])
    for column in ['sex', 'localization', 'confirmation']:
        encoder = OneHotEncoder(sparse=False)
        encoded = encoder.fit_transform(new_dataset_df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        new_dataset_df = pd.concat([new_dataset_df, encoded_df], axis=1)
        new_dataset_df = new_dataset_df.drop(column, axis=1)

    image_ids = new_dataset_df['image_id']

    # Drop the columns 'lesion_type' and 'image_id'
    X = new_dataset_df.drop(['lesion_type', 'image_id'], axis=1)
    y = new_dataset_df['lesion_type']

    scaler = StandardScaler()
    numerical_columns = ['age', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if folder == 'train':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y_encoded)
        y_res = encoder.inverse_transform(y_res)
        new_dataset_df = pd.DataFrame(X_res, columns=X.columns)
        new_dataset_df['lesion_type'] = y_res
    else:
        new_dataset_df = X.copy()
        new_dataset_df['lesion_type'] = y.copy()
    
    if folder == 'test':
        new_dataset_df['image_id'] = image_ids

    new_dataset_df.to_csv(f'dataset/preprocessed_{folder}_dataset.csv', index=False)
    print(f"\nSaved the preprocessed {folder} dataset.")