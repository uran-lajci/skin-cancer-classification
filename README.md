# Skin Cancer Classification

The aim of this project is to develop a machine learning model for skin cancer classification using a dataset comprising of 10015 observations across six variables: lesion type, confirmation method, patient age, gender, and lesion localization. 

## Key Findings from the Exploratory Data Analysis (EDA)

- **Dataset Structure**: The dataset consists of six variables, including both categorical and continuous types. 'Age' is the only continuous variable, while the rest (lesion type, confirmation method, gender, localization) are categorical.

- **Missing Values Handling**: Missing values were present in the 'age' attribute. These were filled using the mean age value, ensuring a complete dataset for model training.

- **Lesion Type Distribution**: Lesion types are not evenly distributed across the dataset. There is a notable class imbalance, with a substantial bias towards the '1' category. This highlights the need for strategies to address class imbalance during model training.

- **Age Distribution**: Patient ages in the dataset are normally distributed, indicating a balanced age distribution among the patients.

- **Confirmation Method**: Among the confirmation methods used, 'histo' and 'follow_up' were most prevalent. This information could potentially be leveraged during model training.

- **Gender Distribution**: Patient gender is almost equally distributed, with a slightly higher proportion of males. Gender could be a relevant feature in predicting lesion type.

- **Lesion Localization**: The most common lesion localizations were 'back' and 'lower extremity'. The location of a lesion may be an important predictor of its type.

- **Statistical Correlations**: Significant correlations were found between 'confirmation' and 'lesion_type', 'sex' and 'lesion_type', and 'localization' and 'lesion_type', suggesting these variables could be informative predictors. However, the Pearson correlation between 'lesion_type' and 'age' was low, indicating a weak linear relationship.

- **Choice of Machine Learning Model**: Given the dataset characteristics, suitable machine learning models for the classification task might include decision trees, random forests, gradient boosting, or neural network-based classifiers.

- **Data Pre-processing**: As part of the data pre-processing steps, encoding will be needed for categorical variables, and normalization or scaling will be required for the continuous variable 'age'. Additionally, sampling strategies may be employed to handle the observed class imbalance.

The insights from the EDA will be pivotal in the subsequent phases of the machine learning project, guiding the choice of machine learning model, the data pre-processing steps, and the strategy for handling class imbalance. This comprehensive understanding of the dataset will ultimately enhance the performance of the machine learning model in the skin cancer classification task.

## Data Preprocessing

1. **Feature Extraction:** For each image in the dataset, we calculate texture features using the Gray Level Co-occurrence Matrix (GLCM). The GLCM is a statistical method that examines the spatial relationship of pixels. It calculates the frequency of a pixel with intensity i being adjacent to a pixel with intensity j. The GLCM features that we calculate are contrast, dissimilarity, homogeneity, energy, correlation, and ASM (angular second moment). These features provide information about the texture of the image, which can be useful in distinguishing different types of skin lesions.

2. **Metadata Preprocessing:** For the metadata, missing 'age' values are filled in with the mean age value. The preprocessing also includes one-hot encoding categorical variables ('sex', 'localization', 'confirmation') to convert them into a format that can be used by machine learning algorithms. We concatenate these newly created one-hot encoded columns to the original dataset and drop the original categorical columns.

3. **Data Scaling:** The numerical features (age, contrast, dissimilarity, homogeneity, energy, correlation, and ASM) are scaled using StandardScaler from sklearn, which standardizes features by removing the mean and scaling to unit variance. Standardization can be crucial for many machine learning algorithms.

4. **Class Imbalance Handling (Only for Training Data):** The script handles the class imbalance problem in the training dataset using SMOTE (Synthetic Minority Over-sampling Technique). SMOTE creates synthetic observations of the minority class by introducing small variations into copies of the existing minority instances. After SMOTE, we have a balanced training dataset, which can improve the performance of the subsequent machine learning model.

5. **Data Saving:** Finally, each preprocessed dataset is saved as a new CSV file, which will be used for training, testing, and validating the machine learning model.

## Model Training

This script random_forest.py trains a Random Forest Classifier on a skin lesion dataset to predict different lesion types. The data is split into features and targets for both training and validation sets. The model is then trained using the training data and predictions are made on the validation set.

Performance metrics including accuracy, precision, recall, and F1-score are calculated to evaluate the model. These metrics provide an understanding of the model's performance on correctly predicting the various skin lesion types.

Finally, the trained model is saved using joblib, enabling it to be used for future predictions on new data.

## Model Inference

This script, `model_inference.py`, conducts the inference process on the preprocessed test dataset using a trained Random Forest Classifier. The key steps are as follows:

1. **Data Preparation**: The test dataset is loaded and separated into image IDs and features. The image IDs are saved for the final output, while the features are used for prediction.

2. **Model Loading**: The pre-trained Random Forest Classifier is loaded into memory using the `joblib` library. This model has been previously trained on skin lesion images and can predict the type of skin lesion in new images.

3. **Prediction**: The model is applied to the features of the test dataset, predicting a numerical label for the lesion type in each image.

4. **Decoding Predictions**: The numerical predictions are then converted back into their original string format ("BKL", "NV", "DF", "MEL", "VASC", "BCC", "AKIEC") using a mapping dictionary.

5. **Output Preparation**: The output is prepared as a list of dictionaries. Each dictionary corresponds to an image in the test dataset and contains the image ID and the predicted lesion type.

6. **Output Saving**: Finally, the output list is saved to a JSON file. This file can be used to check the model's predictions for each image in the test dataset.
