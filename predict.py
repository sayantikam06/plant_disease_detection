"""
predict.py
==========
Handles loading the trained model and making predictions on new leaf images.

This module is imported by app.py (Flask server).
It keeps prediction logic separate from the web server logic — clean design.

PREDICTION PIPELINE:
    Raw image file
        ↓
    Read & decode (OpenCV)
        ↓
    Resize to 224x224
        ↓
    Normalize pixels [0,255] → [0,1]
        ↓
    Add batch dimension: (224,224,3) → (1,224,224,3)
        ↓
    Pass through MobileNetV2 model
        ↓
    Softmax output: [0.85, 0.10, 0.05, ...]
        ↓
    Pick highest probability → class label + confidence score
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "plant_disease_model.h5"
LABELS_PATH = "class_labels.json"
IMG_SIZE    = (224, 224)
# ─────────────────────────────────────────────────────────────────────────────


class PlantDiseasePredictor:
    """
    Singleton-style predictor class.
    We load the model ONCE when Flask starts — not on every request.
    Loading a model is expensive (~2-3 seconds); prediction itself is fast (~0.1s).
    """

    def __init__(self):
        self.model = None
        self.class_labels = None   # dict: {"0": "Healthy", "1": "Leaf_Blight", ...}
        self._load_model()

    def _load_model(self):
        """
        Loads the trained .h5 model and class label mappings from disk.
        Called once when the predictor is instantiated.
        """
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at '{MODEL_PATH}'.\n"
                f"Please run 'python train_model.py' first to train the model."
            )

        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(
                f"Class labels file not found at '{LABELS_PATH}'.\n"
                f"This file is auto-generated during training."
            )

        print(f"[Predictor] Loading model from {MODEL_PATH}...")
        # Load the full Keras model (architecture + weights)
        self.model = tf.keras.models.load_model(MODEL_PATH)

        print(f"[Predictor] Loading class labels from {LABELS_PATH}...")
        with open(LABELS_PATH, "r") as f:
            self.class_labels = json.load(f)  # {"0": "Healthy", "1": "Mildew", ...}

        print(f"[Predictor] Ready! {len(self.class_labels)} classes loaded.")

    def preprocess_image(self, image_path):
        """
        Reads an image from disk and prepares it for the CNN model.

        Steps:
        1. Read with OpenCV (loads as BGR by default)
        2. Convert BGR → RGB (model was trained on RGB images)
        3. Resize to 224×224 (required by MobileNetV2)
        4. Normalize: divide by 255 to get values in [0.0, 1.0]
        5. Add batch dimension: shape (224,224,3) → (1,224,224,3)
           The model expects a batch even for single images

        Returns:
            numpy array of shape (1, 224, 224, 3)
        """
        # Read image from disk
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at path: {image_path}")

        # Convert BGR (OpenCV default) → RGB (what model was trained on)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to 224×224 — MobileNetV2's required input size
        img = cv2.resize(img, IMG_SIZE)

        # Normalize: scale pixel values from [0, 255] to [0.0, 1.0]
        # This matches the rescale=1/255 used in ImageDataGenerator during training
        img = img.astype(np.float32) / 255.0

        # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
        # TensorFlow models always expect a batch as the first dimension
        img = np.expand_dims(img, axis=0)

        return img

    def predict(self, image_path):
        """
        Main prediction function. Takes an image path and returns the disease prediction.

        Args:
            image_path (str): Path to the uploaded leaf image

        Returns:
            dict: {
                "disease"     : "Tomato_Leaf_Blight",   # Predicted class name
                "confidence"  : 91.5,                    # Confidence in percent
                "all_predictions": [                      # Full probability distribution
                    {"disease": "Healthy", "confidence": 5.2},
                    {"disease": "Leaf_Blight", "confidence": 91.5},
                    ...
                ]
            }
        """
        # Step 1: Preprocess the image
        processed_img = self.preprocess_image(image_path)

        # Step 2: Run the image through the CNN model
        # model.predict() returns a 2D array: shape (1, num_classes)
        # e.g., [[0.05, 0.915, 0.035]] for 3 classes
        predictions = self.model.predict(processed_img, verbose=0)

        # predictions[0] = the single image's probability array
        prob_array = predictions[0]  # shape: (num_classes,)

        # Step 3: Find the class with highest probability
        predicted_index = int(np.argmax(prob_array))
        confidence = float(np.max(prob_array)) * 100  # Convert to percentage

        # Look up the class name using the index
        # class_labels keys are strings (from JSON), so convert int → str
        predicted_disease = self.class_labels.get(str(predicted_index), "Unknown")

        # Step 4: Build the full prediction breakdown (sorted by confidence)
        all_preds = []
        for idx, prob in enumerate(prob_array):
            disease_name = self.class_labels.get(str(idx), f"Class_{idx}")
            all_preds.append({
                "disease": disease_name,
                "confidence": round(float(prob) * 100, 2)
            })

        # Sort from highest to lowest confidence
        all_preds.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "disease": predicted_disease,
            "confidence": round(confidence, 2),
            "all_predictions": all_preds
        }


# ── Singleton instance ────────────────────────────────────────────────────────
# Created once when this module is imported by app.py
# This avoids reloading the model on every API request
_predictor_instance = None


def get_predictor():
    """Returns the shared predictor instance (creates it on first call)."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = PlantDiseasePredictor()
    return _predictor_instance
