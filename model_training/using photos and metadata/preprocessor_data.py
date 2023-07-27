import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("new_dataset.csv")

# One-hot encode categorical variables
encoder_sex = OneHotEncoder(sparse=False)
sex_encoded = encoder_sex.fit_transform(df[['sex']])

encoder_localization = OneHotEncoder(sparse=False)
localization_encoded = encoder_localization.fit_transform(df[['localization']])

# Convert arrays to DataFrame
sex_encoded_df = pd.DataFrame(sex_encoded, columns=encoder_sex.get_feature_names_out(['sex']))
localization_encoded_df = pd.DataFrame(localization_encoded, columns=encoder_localization.get_feature_names_out(['localization']))


# Drop original categorical columns
df = df.drop(['sex', 'localization'], axis=1)

# Concatenate one-hot encoded columns
df = pd.concat([df, sex_encoded_df, localization_encoded_df], axis=1)

# Now your data should be preprocessed and ready for prediction
df.to_csv('preprocessed_new_dataset.csv', index=False)