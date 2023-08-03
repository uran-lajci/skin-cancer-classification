import pandas as pd
import warnings

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

warnings.filterwarnings("ignore")


def load_metadata(file_path):
    """Loads and preprocesses metadata"""
    metadata = pd.read_csv(file_path)
    metadata['image_id'] = metadata['image_id'].apply(lambda x: f'{x}.jpg')
    metadata["lesion_type"] = metadata["lesion_type"].apply(lambda x: f'{x}')
    return metadata


def get_datagen():
    """Creates and returns an ImageDataGenerator"""
    datagen = ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.2,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)
    return datagen


def get_generator(datagen, dataframe, directory, subset):
    """Creates and returns a generator for training or validation data"""
    generator = datagen.flow_from_dataframe(
        dataframe=dataframe,
        directory=directory,
        x_col="image_id",
        y_col="lesion_type",
        subset=subset,
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64,64))
    return generator


def build_model():
    """Builds and returns the CNN model"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax')) 
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


if __name__ == "__main__":
    metadata = load_metadata("dataset/metadata.csv")
    datagen = get_datagen()

    train_generator = get_generator(datagen, metadata, "dataset/images/train", "training")
    valid_generator = get_generator(datagen, metadata, "dataset/images/validation", "validation")

    model = build_model()

    model.fit(train_generator,
          validation_data=valid_generator,
          steps_per_epoch=train_generator.n//train_generator.batch_size,
          validation_steps=valid_generator.n//valid_generator.batch_size,
          epochs=10)

    model.save("model_training/models/cnn_model.h5")
    print("Saved the cnn model.")
