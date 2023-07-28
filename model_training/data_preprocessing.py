import os
import cv2
import warnings
import numpy as np
import pandas as pd

from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings("ignore")


def get_unbalanced_choice():
    while True:
        unbalanced_choice = input("""
        Write smote for SMOTE,
        write roc for random oversampling,
        write no_balance if you do no what to tackle the unbalanced dataset.
        Choice: """)

        if unbalanced_choice in ["smote", "roc", "no_balance"]:
            return unbalanced_choice
        else:
            print(f"You wrote a wrong input \"{unbalanced_choice}\" for the choice to tackle the unbalanced dataset. Please try again.")


def calculate_features(image):
    glcm = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    return [np.mean(graycoprops(glcm, prop)) for prop in props]


def get_images_and_calculate_features(image_folder_path):
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


def process_datasets(datasets, metadata, unbalanced_choice):
    image_ids = metadata['image_id']

    for folder, images in datasets.items():
        new_dataset_df = build_and_encode_dataframe(images, metadata)
        X, y = drop_columns_and_scale(new_dataset_df)

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        
        if folder == 'train':
            X, y = handle_imbalance(X, y_encoded, unbalanced_choice)
            new_dataset_df = pd.DataFrame(X, columns=X.columns)
            new_dataset_df['lesion_type'] = encoder.inverse_transform(y)
        else:
            new_dataset_df = X.copy()
            new_dataset_df['lesion_type'] = y.copy()
            new_dataset_df['image_id'] = image_ids
        
        new_dataset_df.to_csv(f'dataset/{unbalanced_choice}_preprocessed_{folder}_dataset.csv', index=False)
        print(f"\nSaved the preprocessed {folder} dataset.")


def build_and_encode_dataframe(images, metadata):
    image_ids, features = zip(*images)
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


def handle_imbalance(X, y_encoded, unbalanced_choice):
    if unbalanced_choice == 'smote':
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y_encoded)
    elif unbalanced_choice == 'roc':
        ros = RandomOverSampler(random_state=42)  
        X_res, y_res = ros.fit_resample(X, y_encoded)
    else:
        X_res, y_res = X, y_encoded
    return X_res, y_res
    

if __name__ == "__main__":
    metadata = pd.read_csv('dataset/metadata.csv')
    metadata['age'].fillna(metadata['age'].mean(), inplace = True)
    
    dataset_folders = ['train', 'test', 'validation']
    datasets = {folder: get_images_and_calculate_features(os.path.join('dataset\\images\\', folder)) for folder in dataset_folders}

    unbalanced_choice = get_unbalanced_choice()
    process_datasets(datasets, metadata, unbalanced_choice)