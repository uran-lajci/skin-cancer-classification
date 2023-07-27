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
