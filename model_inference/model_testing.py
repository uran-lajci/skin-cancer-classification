import pandas as pd
import joblib
import json

# Load test data
test_df = pd.read_csv("dataset/preprocessed_test_dataset.csv")

# Separate features and targets
x_test = test_df.drop(['image_id', 'lesion_type'], axis = 1)
image_ids = test_df['image_id']

# Load the model
rfc = joblib.load('model_training/random_forest_model.pt')

# Run the model on the test set
y_pred = rfc.predict(x_test)

# Create a mapping dictionary
lesion_type_mapping = {
    0: "BKL",
    1: "NV",
    2: "DF",
    3: "MEL",
    4: "VASC",
    5: "BCC",
    6: "AKIEC"
}

# Create a list of dictionaries for the output
output = []
for image_id, lesion_type in zip(image_ids, y_pred):
    output.append({
        "image_id": str(image_id),  # If image_id is a numerical type, convert to string
        "lesion_type": lesion_type_mapping[int(lesion_type)]  # Use the mapping dictionary to get the lesion type string
    })

# Save the output to a JSON file
with open("model_inference/output.json", "w") as f:
    json.dump(output, f)

print("Completed the model inference.")