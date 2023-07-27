import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load training data
train_df = pd.read_csv("dataset/preprocessed_train_dataset.csv")

# Load validation data
valid_df = pd.read_csv("dataset/preprocessed_validation_dataset.csv")

# Separate features and targets
x_train = train_df.drop(['lesion_type'], axis = 1)
y_train = train_df['lesion_type']

x_valid = valid_df.drop(['lesion_type'], axis = 1)
y_valid = valid_df['lesion_type']

if 'localization_acral' not in x_valid.columns:
    x_valid['localization_acral'] = 0.0

x_valid = x_valid.reindex(columns = x_train.columns)

# Initialize the model
rfc = RandomForestClassifier(random_state=42)

# Train the model
rfc.fit(x_train, y_train)

# Validate the model using validation set
y_pred = rfc.predict(x_valid)

# Calculate the evaluation metrics
accuracy = accuracy_score(y_valid, y_pred)
precision = precision_score(y_valid, y_pred, average='weighted')
recall = recall_score(y_valid, y_pred, average='weighted')
f1 = f1_score(y_valid, y_pred, average='weighted')

print('Evaluation of Random Forest')
print('Accuracy', accuracy)
print('Precision', precision)
print('Recall', recall)
print('F1-Score', f1)

# Save the model
joblib.dump(rfc, 'model_training/random_forest_model.pt')
print("Saved the random forest model.")