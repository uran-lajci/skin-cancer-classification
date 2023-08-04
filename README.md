# Skin Cancer Classification

The aim of this project is to develop a machine learning model for skin cancer classification using a dataset comprising of 10015 observations across six variables: lesion type, confirmation method, patient age, gender, and lesion localization. 

## Key Findings from the Exploratory Data Analysis (EDA)

- **Dataset Structure**: The dataset consists of six variables, including both categorical and continuous types. 'Age' is the only continuous variable, while the rest (lesion type, confirmation method, gender, localization) are categorical.

- **Missing Values Handling**: Missing values were present in the 'age' attribute. (These will be handled in the preprocessing script).

- **Lesion Type Distribution**: Lesion types are not evenly distributed across the dataset. There is a notable class imbalance, with a substantial bias towards the '1' category. This highlights the need for strategies to address class imbalance during model training.

- **Age Distribution**: Patient ages in the dataset are normally distributed, indicating a balanced age distribution among the patients.

- **Confirmation Method**: Among the confirmation methods used, 'histo' and 'follow_up' were most prevalent. This information could potentially be leveraged during model training.

- **Gender Distribution**: Patient gender is almost equally distributed, with a slightly higher proportion of males. Gender could be a relevant feature in predicting lesion type.

- **Lesion Localization**: The most common lesion localizations were 'back' and 'lower extremity'. The location of a lesion may be an important predictor of its type.

- **Chi-square Correlations**: 
* 'confirmation' and 'lesion_type': A significant association is found (p-value = 0.0), indicating that the confirmation method might be related to the lesion type.
* 'sex' and 'lesion_type': A significant association is found (p-value < 0.001), showing that the sex of the patient might be connected to the lesion type.
* 'localization' and 'lesion_type': A significant association (p-value = 0.0) suggests that the localization of the lesion may be related to its type.
* 'confirmation' and 'sex' with 'localization': A significant association (p-value = 0.0) indicates that the combination of confirmation and sex might be related to the localization of the lesion.

- **Choice of Machine Learning Model**: Given the dataset characteristics, suitable machine learning models for the classification task might include decision trees, random forests, gradient boosting, or neural network-based classifiers.

- **Data Pre-processing**: As part of the data pre-processing steps, encoding will be needed for categorical variables, and normalization or scaling will be required for the continuous variable 'age'. Additionally, sampling strategies may be employed to handle the observed class imbalance.

The insights from the EDA will be pivotal in the subsequent phases of the machine learning project, guiding the choice of machine learning model, the data pre-processing steps, and the strategy for handling class imbalance. This comprehensive understanding of the dataset will ultimately enhance the performance of the machine learning model in the skin cancer classification task.

## Data Preprocessing

1. **Feature Extraction**: For each image in the dataset, the script calculates texture features using the Gray Level Co-occurrence Matrix (GLCM). The GLCM is a statistical method that examines the spatial relationship of pixels in the grayscale images. It calculates the frequency of a pixel with intensity i being adjacent to a pixel with intensity j. The GLCM features computed for each image are contrast, dissimilarity, homogeneity, energy, correlation, and ASM (angular second moment). These features provide information about the texture of the image, which can be useful in distinguishing different types of skin lesions.

2. **Metadata Preprocessing**: The script handles missing 'age' values in the metadata by filling them in with the mean age value of the available data. Categorical variables ('sex', 'localization', 'confirmation') in the metadata are one-hot encoded to convert them into a format suitable for machine learning algorithms. The newly created one-hot encoded columns are then concatenated to the original dataset, and the original categorical columns are dropped.

3. **Data Scaling**: The numerical features (age, contrast, dissimilarity, homogeneity, energy, correlation, and ASM) in the dataset are scaled using the StandardScaler from the sklearn library. StandardScaler standardizes features by removing the mean and scaling them to have unit variance. This step is crucial for many machine learning algorithms to ensure that all features contribute equally to the model's performance.

4. **Class Imbalance Handling (Only for Training Data)**: The script addresses the class imbalance problem in the training dataset using either SMOTE (Synthetic Minority Over-sampling Technique) or Random Over-sampling. If the user selects "smote," the script applies SMOTE to create synthetic observations of the minority class by introducing small variations into copies of the existing minority instances. If "roc" is chosen, the script uses Random Over-sampling to randomly duplicate samples from the minority class to balance the dataset. If the user selects "no_balance," the script does not handle the class imbalance issue, and the dataset remains unbalanced.

5. **Data Saving**: Finally, after processing each dataset (train, test, validation), the preprocessed data is saved as a new CSV file. The file name includes the method used for class imbalance handling (if applicable) to indicate which preprocessing steps were applied. The saved preprocessed datasets will be used for training, testing, and validating the machine learning model.

Note: The script reads the metadata from a CSV file named 'metadata.csv' and the images from the respective folders ('train', 'test', 'validation') within the 'dataset/images/' directory. The user is prompted to choose the method for handling class imbalance, and based on the choice, the script performs the necessary data preprocessing steps.

## Model Training

The script `rfc_and_dtc.py` and `cnn.py` perform training of machine learning models for predicting skin lesion types. The data preprocessing steps are assumed to be completed beforehand using the `data_preprocessing.py` script.

1. **rfc_and_dtc.py**: This script allows you to choose a dataset version (`no_balance`, `smote`, or `roc`) and the machine learning algorithm (`RFC` for Random Forest Classifier or `DTC` for Decision Tree Classifier). It then loads the preprocessed datasets, splits them into features and targets for training and validation sets, builds and fits the selected model, and finally evaluates its performance using various metrics. The trained model is saved using the joblib library for future use.

2. **cnn.py**: This script uses convolutional neural networks (CNNs) to train a model for image classification. It loads the metadata, preprocesses it by converting image IDs to the appropriate format, and creates an ImageDataGenerator with data augmentation settings. The script then generates data flow for both training and validation datasets using the ImageDataGenerator. The CNN model is defined, compiled, and trained on the generated data. The trained CNN model is then saved in the h5 format.

## Model Inference

The script `model_inference.py` conducts the inference process on the preprocessed test dataset using a trained machine learning model, which can either be a Random Forest Classifier (RFC) or a Decision Tree Classifier (DTC). The key steps are as follows:

1. **Data Preparation**: The test dataset is loaded from the file "dataset/no_balance_preprocessed_test_dataset.csv." The features (i.e., image attributes) are separated from the dataset, and the corresponding image IDs are saved for the final output. These features will be used for prediction.

2. **Model Loading**: The script prompts the user to choose either "RFC" for Random Forest Classifier or "DTC" for Decision Tree Classifier. The pre-trained model of the selected algorithm is then loaded into memory using the `joblib` library. The model has been previously trained on skin lesion images and can predict the type of skin lesion in new images.

3. **Prediction**: The model is applied to the features of the test dataset, predicting a numerical label for the lesion type in each image.

4. **Decoding Predictions**: The numerical predictions are then converted back into their original string format ("BKL", "NV", "DF", "MEL", "VASC", "BCC", "AKIEC") using a mapping dictionary.

5. **Output Preparation**: The output is prepared as a list of dictionaries. Each dictionary corresponds to an image in the test dataset and contains the image ID and the predicted lesion type in string format.

6. **Output Saving**: Finally, the output list is saved to a JSON file named "{algorithm_choice}_output.json" in the "model_inference" directory. This JSON file can be used to check the model's predictions for each image in the test dataset.

## Results

**CNN Resulsts**

![image](https://github.com/uran-lajci/skin-cancer-classification/assets/117693854/54428c4e-f4ab-4d81-be7a-a3b39a19371c)

**Decision Tree Classifier and Random Forest Classifier Results**

![image](https://github.com/uran-lajci/skin-cancer-classification/assets/117693854/e4fd4b1d-e266-4ae0-9d8a-b1264875000f)

## Resources

- https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- https://github.com/bundasmanu/skin_mnist
- https://www.nature.com/articles/s41598-022-22644-9
- https://www.apriorit.com/dev-blog/647-ai-applying-deep-learning-to-classify-skin-cancer-types
- https://www.youtube.com/watch?v=nVhau51w6dM
- https://www.youtube.com/playlist?list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF
- Chat GPT (Version 3.5 and 4)
- Stack Overflow