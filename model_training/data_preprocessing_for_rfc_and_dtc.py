import os
import cv2
import warnings
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")


def get_the_imbalanced_technique_choice():
    while True:
        imbalanced_technique = input("""
        Write smote for SMOTE,
        write roc for random oversampling,
        write no_balance if you do no what to tackle the unbalanced dataset.
        Choice: """)

        if imbalanced_technique in ["smote", "roc", "no_balance"]:
            return imbalanced_technique
        else:
            print(f"You wrote a wrong input \"{imbalanced_technique}\" for the choice to tackle the unbalanced dataset. Please try again.")


def calculate_features(image):
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    return [np.mean(graycoprops(glcm, prop)) for prop in props]


def images_with_extracted_features(image_folder_path):
    images = []
    for image_name in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_name)
        image_id = image_name.split('.')[0]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        features = calculate_features(image)
        images.append((image_id, features))
        if len(images) % 100 == 0:
            print(f"Extracted features for {len(images)} images in {image_folder_path}.")
    return images


def process_datasets(folders_with_image_features, metadata, imbalanced_technique):
    image_ids = metadata['image_id']

    for folder, image_ids_with_features in folders_with_image_features.items():
        df = build_and_encode_dataframe(image_ids_with_features, metadata)
        X, y = drop_columns_and_scale(df)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        if folder == 'train':
            X, y = handle_imbalance(X, y_encoded, imbalanced_technique)
            df = pd.DataFrame(X, columns=X.columns)
            df['lesion_type'] = encoder.inverse_transform(y)
            df.to_csv(f'dataset/{imbalanced_technique}_preprocessed_{folder}_dataset.csv', index=False)

        elif folder == 'validation':
            if not os.path.isfile('dataset/preprocessed_validation_dataset.csv'):
                df = X.copy()
                df['lesion_type'] = y.copy()
                df['image_id'] = image_ids
                df.to_csv(f'dataset/preprocessed_validation_dataset.csv', index=False)
        
        elif folder == 'test':
            if not os.path.isfile('dataset/preprocessed_test_dataset.csv'):
                df = X.copy()
                df['lesion_type'] = y.copy()
                df['image_id'] = image_ids
                df.to_csv(f'dataset/preprocessed_test_dataset.csv', index=False)
        
        print(f"\nSaved the preprocessed {folder} dataset.")


def build_and_encode_dataframe(image_ids_with_features, metadata):
    image_ids, features = zip(*image_ids_with_features)
    contrast, dissimilarity, homogeneity, energy, correlation, asm = zip(*features)
    df = pd.DataFrame({
        'image_id': image_ids, 
        'contrast': contrast, 
        'dissimilarity': dissimilarity, 
        'homogeneity': homogeneity, 
        'energy': energy, 
        'correlation': correlation, 
        'asm': asm
    })
    df = pd.concat([metadata.loc[metadata['image_id'].isin(df['image_id'])].reset_index(drop=True), df], axis=1)
    for column in ['sex', 'localization', 'confirmation']:
        encoder = OneHotEncoder(sparse=False)
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        df = pd.concat([df, encoded_df], axis=1)
        df = df.drop(column, axis=1)
    return df


def drop_columns_and_scale(df):
    X = df.drop(['lesion_type', 'image_id'], axis=1)
    y = df['lesion_type']
    scaler = StandardScaler()
    numerical_columns = ['age', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
    return X, y


def handle_imbalance(X, y, imbalance_technique):
    if imbalance_technique == 'smote':
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
    
    if imbalance_technique == 'roc':
        ros = RandomOverSampler(random_state=42)  
        return ros.fit_resample(X, y)
    
    return X, y
    
if __name__ == "__main__":
    metadata = pd.read_csv('dataset/metadata.csv')

    # handles the missing age data
    metadata['age'].fillna(metadata['age'].mean(), inplace = True)
    
    dataset_folders = ['train', 'test', 'validation']
    folders_with_image_features = {folder: images_with_extracted_features(os.path.join('dataset\\images\\', folder)) for folder in dataset_folders}

    process_datasets(folders_with_image_features, metadata, get_the_imbalanced_technique_choice())