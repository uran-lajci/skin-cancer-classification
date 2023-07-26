from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

import pandas as pd

pd.set_option('display.max_columns', None)

# data read
df = pd.read_csv("./dataset/metadata.csv")

df = df.drop(columns=['image_id'])

# Perform one-hot encoding on categorical variables
df = pd.get_dummies(df, columns=['confirmation', 'sex', 'localization'])

# Separate the training and testing sets based on 'lesion_type'
train_df = df[df['lesion_type'] != -1]

# We see from your DataFrame's info that there are some missing values in 'age', we'll fill them with the median
train_df.loc[:, 'age'] = train_df['age'].fillna(train_df['age'].median())

# Define the features and target variable for your model
X = train_df.drop('lesion_type', axis=1)
y = train_df['lesion_type']

# Split the dataset into a training set and a validation set
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Save the trained model to a file
dump(clf, f'model_training/using only metadata/RFC_metadata_model')

# Use the model to make predictions on the validation set
y_pred = clf.predict(X_val)

# Print the performance metrics
print(classification_report(y_val, y_pred))