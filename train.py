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
import kagglehub

def clean_label(label):
    label = label.lower()
    if 'adenocarcinoma' in label:
        return 'Adenocarcinoma'
    elif 'large.cell.carcinoma' in label:
        return 'Large Cell Carcinoma'
    elif 'squamous.cell.carcinoma' in label:
        return 'Squamous Cell Carcinoma'
    else:
        return 'Normal'

def get_dataframe(data_dir, split_name):
    filepaths = []
    labels = []
    split_path = os.path.join(data_dir, split_name)
    if not os.path.exists(split_path):
        return pd.DataFrame()

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                filepaths.append(os.path.join(class_path, img_file))
                labels.append(clean_label(class_name))
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

def train():
    print("Downloading dataset...")
    path = kagglehub.dataset_download('mohamedhanyyy/chest-ctscan-images')
    data_dir = os.path.join(path, "Data")

    train_df = get_dataframe(data_dir, 'train')
    valid_df = get_dataframe(data_dir, 'valid')

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Using ResNet50V2 specific preprocessing
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.15,
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
        class_mode='categorical'
    )

    # Handle Class Imbalance
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_arr))
    print(f"Computed Class Weights: {class_weights}")

    num_classes = train_df['label'].nunique()

    base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    ]

    print("Training top layers...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=15,
        class_weight=class_weights,
        callbacks=callbacks
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    print("Fine tuning model...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=30,
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
