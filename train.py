import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import kagglehub

def train():
    print("Downloading dataset...")
    path = kagglehub.dataset_download('mohamedhanyyy/chest-ctscan-images')
    data_dir = os.path.join(path, "Data")

    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')

    # Increased image size for better feature extraction (GPU optimized)
    IMG_SIZE = (300, 300)
    BATCH_SIZE = 32

    # EfficientNet expects inputs in range [0, 255] and handles scaling internally.
    # We apply robust data augmentation here
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    valid_generator = test_datagen.flow_from_directory(
        valid_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)

    # Use EfficientNetB3 for higher capacity
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(300, 300, 3))

    # Freeze the base model
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x) # Stronger dropout
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    callbacks = [
        ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
    ]

    # Initial training of top layers
    print("Training top layers...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=15, # More epochs
        callbacks=callbacks
    )

    # Fine-tuning: unfreeze some top layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]: # Unfreeze last 30 layers
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=1e-5), # Very low learning rate for fine tuning
                  loss='categorical_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    print("Fine tuning model...")
    model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=30, # More epochs for fine tuning
        callbacks=callbacks
    )

    print("Training Complete. Model saved as best_model.h5.")

if __name__ == '__main__':
    # Optional: configure TensorFlow to allocate GPU memory dynamically
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

    train()
