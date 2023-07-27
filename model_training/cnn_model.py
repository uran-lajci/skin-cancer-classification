import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load metadata
metadata = pd.read_csv("dataset/metadata.csv")

metadata['image_id'] = metadata['image_id'].apply(lambda x: f'{x}.jpg')
metadata["lesion_type"] = metadata["lesion_type"].apply(lambda x: f'{x}')

# Prepare ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255.,
    validation_split=0.2,
    rotation_range=20,       # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.2,          # randomly zoom image 
    width_shift_range=0.2,   # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,    # randomly flip images
    vertical_flip=True)      # randomly flip images

train_generator = datagen.flow_from_dataframe(
    dataframe=metadata,
    directory="dataset/images/train",
    x_col="image_id",
    y_col="lesion_type",
    subset="training",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

valid_generator = datagen.flow_from_dataframe(
    dataframe=metadata,
    directory="dataset/images/validation",
    x_col="image_id",
    y_col="lesion_type",
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(64,64))

# Define a CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # 7 classes

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_generator,
          validation_data=valid_generator,
          steps_per_epoch=train_generator.n//train_generator.batch_size,
          validation_steps=valid_generator.n//valid_generator.batch_size,
          epochs=10)

# Save the model
model.save("model_train/cnn_model.h5")