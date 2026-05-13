import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import kagglehub

def get_dataframe(data_dir):
    """Recursively fetch all image files and return a DataFrame with paths and labels."""
    filepaths = []
    labels = []

    # Kaggle dataset often has train/test/valid folders. We'll pool them all for k-fold
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for img_file in os.listdir(class_path):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(class_path, img_file))
                    labels.append(class_name)

    df = pd.DataFrame({
        'filepath': filepaths,
        'label': labels
    })
    return df

def build_model(num_classes):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def train_kfold():
    print("Downloading dataset...")
    path = kagglehub.dataset_download('mohamedhanyyy/chest-ctscan-images')
    data_dir = os.path.join(path, "Data")

    df = get_dataframe(data_dir)
    print(f"Total images found: {len(df)}")

    num_classes = df['label'].nunique()
    print(f"Classes: {df['label'].unique()}")

    IMG_SIZE = (300, 300)
    BATCH_SIZE = 32
    FOLDS = 5

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    valid_datagen = ImageDataGenerator()

    best_overall_val_acc = 0.0
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['filepath'], df['label'])):
        print(f"\n--- Starting Fold {fold + 1}/{FOLDS} ---")

        train_df = df.iloc[train_idx]
        valid_df = df.iloc[val_idx]

        train_generator = train_datagen.flow_from_dataframe(
            train_df,
            x_col='filepath',
            y_col='label',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        valid_generator = valid_datagen.flow_from_dataframe(
            valid_df,
            x_col='filepath',
            y_col='label',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False
        )

        model, base_model = build_model(num_classes)

        # Phase 1: Top layers
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
        ]

        print("Training top layers...")
        model.fit(train_generator, validation_data=valid_generator, epochs=10, callbacks=callbacks)

        # Phase 2: Fine-Tuning
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False

        model.compile(optimizer=Adam(learning_rate=1e-5),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Best model path specific to this fold
        fold_model_path = f'model_fold_{fold+1}.h5'
        callbacks.append(ModelCheckpoint(fold_model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=0))

        print("Fine tuning model...")
        history = model.fit(train_generator, validation_data=valid_generator, epochs=20, callbacks=callbacks)

        # Check if this fold produced the overall best model
        val_accs = history.history.get('val_accuracy', [0])
        best_fold_acc = max(val_accs)

        if best_fold_acc > best_overall_val_acc:
            best_overall_val_acc = best_fold_acc
            best_fold = fold + 1
            # Copy to global best model
            if os.path.exists(fold_model_path):
                import shutil
                shutil.copy(fold_model_path, 'best_model.h5')
                print(f"--> New best overall model found! (Val Acc: {best_overall_val_acc:.4f}) Saved as best_model.h5")

    print(f"\nK-Fold Training Complete. Best model was from Fold {best_fold} with Validation Accuracy: {best_overall_val_acc:.4f}.")

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
        print("No GPU found, falling back to CPU.")

    train_kfold()
