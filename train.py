import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import kagglehub

def get_dataframe(data_dir):
    """
    The COVID-19 Radiography Database has folders for each class directly inside the main folder.
    Classes: COVID, Lung_Opacity, Normal, Viral Pneumonia
    """
    filepaths = []
    labels = []

    # We navigate into the main COVID-19_Radiography_Dataset folder
    main_folder = os.path.join(data_dir, "COVID-19_Radiography_Dataset")
    if not os.path.exists(main_folder):
        # Fallback if structure is different
        main_folder = data_dir

    for class_name in os.listdir(main_folder):
        class_path = os.path.join(main_folder, class_name)
        if not os.path.isdir(class_path) or class_name.startswith('.'):
            continue

        # The images are usually inside an 'images' subfolder or directly in the class folder
        image_dir = os.path.join(class_path, 'images')
        if not os.path.exists(image_dir):
            image_dir = class_path

        for img_file in os.listdir(image_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(image_dir, img_file))
                labels.append(class_name.replace('_', ' '))

    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })
    return df

def train():
    print("Downloading massive COVID-19 Radiography Database...")
    path = kagglehub.dataset_download('tawsifurrahman/covid19-radiography-database')

    df = get_dataframe(path)
    print(f"Total images found: {len(df)}")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # Split into train (80%), validation (10%), test (10%)
    train_df, dummy_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    valid_df, test_df = train_test_split(dummy_df, test_size=0.5, stratify=dummy_df['label'], random_state=42)

    # Save test_df to a csv so evaluate.py knows exactly which images were kept out of training
    test_df.to_csv("test_data.csv", index=False)

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filepath',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = test_datagen.flow_from_dataframe(
        valid_df,
        x_col='filepath',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Computed Class Weights: {class_weights}")

    num_classes = len(train_generator.class_indices)
    print(f"Classes: {train_generator.class_indices}")

    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
    ]

    print("Training top layers...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=10,
        class_weight=class_weights,
        callbacks=callbacks
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Fine tuning model...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks
    )
    print("Training Complete. Model saved as best_model.h5.")

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU(s) found: {gpus}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found by TensorFlow.")

    train()
