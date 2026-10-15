import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from sklearn.utils.class_weight import compute_class_weight

# Dataset Directories
train_dir = 'data/train'
val_dir = 'data/val'

# Data Augmentation Setup
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Handle Class Imbalance
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))
print(f"-> Class Imbalance Weights: {class_weights_dict}")

# Setup Training Logs Directory and CSVLogger
os.makedirs('logs', exist_ok=True)
csv_logger = CSVLogger('logs/training_log.csv', append=True)

# Base Model Setup
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# --- PHASE 1: Train Classification Head ---
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

print("\n--- PHASE 1: Training Classification Head ---")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    class_weight=class_weights_dict,
    callbacks=[csv_logger]
)

# --- PHASE 2: Fine-Tuning Deeper Layers ---
print("\n--- PHASE 2: Fine-Tuning ResNet50 Base Layers ---")
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
)

# Callbacks for optimization and logging
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7),
    csv_logger
]

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=12,
    class_weight=class_weights_dict,
    callbacks=callbacks
)

# Save Model
os.makedirs('models', exist_ok=True)
model.save('models/pneumonia_resnet50.keras')
print("\n-> Optimized Model saved successfully to models/pneumonia_resnet50.keras!")