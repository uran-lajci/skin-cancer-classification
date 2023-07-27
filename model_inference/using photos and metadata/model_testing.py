import os
import json
import numpy as np
import pandas as pd
import keras.utils as image
from keras.models import load_model

# Load the trained model
model = load_model('model_training/using photos and metadata/model.h5')

# Load the image filenames
metadata = pd.read_csv('dataset/metadata.csv')

metadata = metadata[metadata['lesion_type'] == -1]

# Function to prepare the image
def prepare_image(file):
    img_path = 'dataset/images/test/'
    img = image.load_img(img_path + file + ".jpg", target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims

def prepare_metadata(image_id):
    meta = metadata[metadata['image_id'] == image_id.replace('.jpg', '')]
    
    # Map the strings to integers
    histo_mapping = {'histo1': 0, 'histo2': 1, 'histo3': 2}  # Replace these with actual categories
    sex_mapping = {'female': 0, 'male': 1}
    localization_mapping = {'back': 0, 'front': 1}  # Replace these with actual categories
    
    histo = histo_mapping[meta['histo'].values[0]]
    age = float(meta['age'].values[0])
    sex = sex_mapping[meta['sex'].values[0]]
    localization = localization_mapping[meta['localization'].values[0]]
    
    meta_values = np.array([[histo, age, sex, localization]], dtype=float)
    
    return meta_values


# List of lesion type indices, replace these with the actual class names
lesion_type_dict = {0: 'AK', 1: 'BCC', 2: 'BKL', 3: 'DF', 4: 'MEL', 5: 'NV', 6: 'VASC'}

# Iterate over images in the metadata, run classifier and collect results
results = []
for idx, row in metadata.iterrows():
    image_id = row['image_id']
    prepared_image = prepare_image(image_id)
    prepared_metadata = prepare_metadata(image_id)
    prediction = model.predict([prepared_image, prepared_metadata])
    lesion_type = lesion_type_dict[np.argmax(prediction)]
    results.append({"image_id": image_id, "lesion_type": lesion_type})

# Save results to a JSON file
with open('model_inference/using photos and metadata/results.json', 'w') as f:
    json.dump(results, f)
