import pandas as pd
from joblib import load
import json

# Define the mapping from numerical values to lesion types
lesion_type_dict = {0: 'BKL', 1: 'NV', 2: 'DF', 3: 'MEL', 4: 'VASC', 5: 'BCC', 6: 'AKIEC'}

# Load the trained model from the file
clf = load('model_training/using only metadata/RFC_metadata_model')

# data read with 'image_id'
df = pd.read_csv("./dataset/metadata.csv")

# Save the image_ids for the test set before dropping the column
image_ids_test = df[df['lesion_type'] == -1]['image_id']

df_encoded = df.drop(columns=['image_id'])

# Perform one-hot encoding on categorical variables
df_encoded = pd.get_dummies(df_encoded, columns=['confirmation', 'sex', 'localization'])

# Separate the testing sets based on 'lesion_type'
test_df = df_encoded[df_encoded['lesion_type'] == -1]

# We see from your DataFrame's info that there are some missing values in 'age', we'll fill them with the median
test_df.loc[:, 'age'] = test_df['age'].fillna(test_df['age'].median())

# Get the features for the test set
X_test = test_df.drop('lesion_type', axis=1)

# Make predictions for the test set
y_pred_test = clf.predict(X_test)

# Convert numerical predictions to lesion types using the defined dictionary
lesion_type_predictions = [lesion_type_dict[pred] for pred in y_pred_test]

# Create a list of dictionaries for the JSON output
json_output = [{"image_id": img_id, "lesion_type": lesion} for img_id, lesion in zip(image_ids_test, lesion_type_predictions)]

# Save JSON output to a file
with open('model_training/using only metadata/test_predictions.json', 'w') as f:
    json.dump(json_output, f)