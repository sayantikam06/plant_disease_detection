"""
train_model.py
==============
Trains a plant disease classifier using MobileNetV2 (Transfer Learning).

HOW IT WORKS:
- MobileNetV2 is a lightweight CNN pre-trained on ImageNet (1.4M images, 1000 classes)
- We "freeze" all its layers (don't change their weights) and add our own classification layers on top
- We only train our new top layers first, then optionally "fine-tune" the top layers of MobileNetV2
- This approach is called Transfer Learning — the model already knows what edges, textures,
shapes look like; we just teach it to distinguish our specific plant disease classes

EXPECTED DATASET STRUCTURE:
    dataset/
        train/
            Healthy/            <- folder name = class label
                img1.jpg
                img2.jpg
            Powdery_Mildew/
                img1.jpg
            Leaf_Blight/
                img1.jpg
        val/
            Healthy/
            Powdery_Mildew/
            Leaf_Blight/

You can use the PlantVillage dataset from Kaggle:
https://www.ka ggle.com/datasets/abdallahalidev/plantvillage-dataset
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20
TRAIN_DIR = 'dataset_split/train'
VAL_DIR = 'dataset_split/val'
MODEL_PATH  = "plant_disease_model.h5"
LABELS_PATH = "class_labels.json"


def build_data_generators():
    """
    Creates training and validation data pipelines.

    Training pipeline includes DATA AUGMENTATION — artificially creating variations
    of each image so the model becomes robust to real-world conditions like:
    - Different angles (rotation, flip)
    - Different lighting (brightness shift)
    - Partial views (zoom, shift)

    Validation pipeline ONLY normalizes (no augmentation) — we want a clean test.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    return train_gen, val_gen


def build_model(num_classes):
    """
    Builds the CNN model using MobileNetV2 as the base (Transfer Learning).

    ARCHITECTURE:
        Input (224x224x3)
            ↓
        MobileNetV2 base (frozen) — extracts features like edges, textures, patterns
            ↓
        GlobalAveragePooling2D — reduces feature maps to a flat 1280-dim vector
            ↓
        Dense(256, ReLU) — learns high-level combinations of features
            ↓
        Dropout(0.5) — randomly turns off 50% of neurons during training
        to prevent overfitting (memorizing instead of learning)
            ↓
        Dense(num_classes, Softmax) — outputs probability for each disease class
    """


    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model, base_model


def fine_tune_model(model, base_model, num_layers_to_unfreeze=30):
    """
    OPTIONAL Fine-Tuning Phase:
    After the top layers converge, we unfreeze the last N layers of MobileNetV2
    and train them at a very low learning rate. This lets the base model
    adapt its features slightly to our specific plant disease domain.

    Too many unfrozen layers + high LR = destroys pre-trained weights (catastrophic forgetting)
    """

    base_model.trainable = True

    for layer in base_model.layers[:-num_layers_to_unfreeze]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train():
    print("=" * 60)
    print("  AI-Based Plant Disease Detection — Model Training")
    print("=" * 60)

    print("\n[1/5] Loading dataset...")
    train_gen, val_gen = build_data_generators()

    num_classes = len(train_gen.class_indices)
    print(f"      Found {num_classes} disease classes: {list(train_gen.class_indices.keys())}")
    print(f"      Training samples  : {train_gen.samples}")
    print(f"      Validation samples: {val_gen.samples}")

    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"      Saved class labels → {LABELS_PATH}")

    print("\n[2/5] Building MobileNetV2 model...")
    model, base_model = build_model(num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    print("\n[3/5] Configuring training callbacks...")

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\n[4/5] Phase 1: Training top layers only (base model frozen)...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    print("\n[5/5] Phase 2: Fine-tuning top layers of MobileNetV2...")
    model = fine_tune_model(model, base_model, num_layers_to_unfreeze=30)

    fine_tune_callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
    ]

    model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=fine_tune_callbacks,
        verbose=1
    )

    print("\n" + "=" * 60)
    print("  Training Complete!")
    print(f"  Best model saved → {MODEL_PATH}")
    print("=" * 60)

    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"  Final Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  Final Validation Loss     : {val_loss:.4f}")


if __name__ == "__main__":
    train()
