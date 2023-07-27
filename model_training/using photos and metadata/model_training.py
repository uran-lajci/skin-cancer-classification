import pandas as pd
from keras import layers,models,utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
import numpy as np

# Load metadata
metadata = pd.read_csv('dataset/metadata.csv')

metadata['image_id'] = metadata['image_id'] + '.jpg'

# Convert numerical labels to string
metadata['lesion_type'] = metadata['lesion_type'].astype(str)

# Encode categorical metadata to numerical
le = preprocessing.LabelEncoder()
metadata['sex'] = le.fit_transform(metadata['sex'])
metadata['localization'] = le.fit_transform(metadata['localization'])
metadata['confirmation'] = le.fit_transform(metadata['confirmation'])

# Normalize age
metadata['age'] = metadata['age'] / 100  # Assuming age range 0-100

# Function to get metadata given image id
def get_metadata(image_id):
    meta = metadata[metadata['image_id'] == image_id]
    return [meta['confirmation'].values[0], meta['age'].values[0], meta['sex'].values[0], meta['localization'].values[0]]

# Custom data generator
class CustomDataGen(utils.Sequence):
    def __init__(self, dataframe, directory, batch_size):
        self.directory = directory
        self.batch_size = batch_size
        self.image_data_gen = ImageDataGenerator(rescale=1./255)
        self.image_gen = self.image_data_gen.flow_from_dataframe(dataframe, directory, x_col="image_id", y_col="lesion_type",
                                                                 target_size=(150, 150), batch_size=batch_size, class_mode='categorical')

    def __len__(self):
        return len(self.image_gen)

    def __getitem__(self, index):
        batch_x, batch_y = self.image_gen[index]
        batch_meta = np.array([get_metadata(id) for id in self.image_gen.filenames[index*self.batch_size:(index+1)*self.batch_size]])
        return [batch_x, batch_meta], batch_y

# Create a simple CNN model with metadata input
def create_model():
    # Image input
    image_input = layers.Input(shape=(150, 150, 3))
    x = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # Metadata input
    meta_input = layers.Input(shape=(4,))
    y = layers.Dense(32, activation='relu')(meta_input)
    
    # Merge inputs
    merged = layers.concatenate([x, y])

    # Classification layer
    outputs = layers.Dense(7, activation='softmax')(merged)

    model = models.Model(inputs=[image_input, meta_input], outputs=outputs)

    return model

# Directory for training and validation images
train_dir = 'dataset/images/train'
validation_dir = 'dataset/images/validation'

# Split the metadata into training and validation
train_metadata = metadata.sample(frac=0.8, random_state=0)  # 80% for training
validation_metadata = metadata.drop(train_metadata.index)  # remaining 20% for validation

# Create Data Generators
train_generator = CustomDataGen(train_metadata, train_dir, batch_size=32)
validation_generator = CustomDataGen(validation_metadata, validation_dir, batch_size=32)

# Create your model
model = create_model()

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, validation_steps=50, epochs=10, validation_data=validation_generator)

# Save the model
model.save('model_training/using photos and metadata/model.h5')