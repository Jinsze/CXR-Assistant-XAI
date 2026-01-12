"""
Model inference utilities for lung disease classification.
Handles model loading and prediction with confidence scoring.
"""

import os
import numpy as np
from typing import Tuple, Optional
import tensorflow as tf

# Import preprocessing functions
from .preprocessing import preprocess_for_inference

# Model configuration
# Point to the actual trained model from Colab
MODEL_PATH = r"C:\Users\asusv\OneDrive\Documents\DSP\ChestXrayProject\models\colab_clahe_eff_final.keras"

class LungDiseaseClassifier:
    """
    Lung disease classification model wrapper.
    Handles model loading, inference, and result formatting.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        """
        Initialize the classifier with a trained model.

        Args:
            model_path: Path to the trained Keras model file
        """
        self.model_path = model_path
        self.model = None
        # IMPORTANT: Class labels MUST match the exact order from training
        # From train_flow.class_indices: {'covid': 0, 'normal': 1, 'pneumonia': 2, 'tuberculosis': 3}
        self.class_labels = [
            "COVID-19",      # Index 0
            "Normal",        # Index 1  
            "Pneumonia",     # Index 2
            "Tuberculosis"   # Index 3
        ]

    def load_model(self) -> bool:
        """
        Load the Keras model from disk.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False

            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict(self, image: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
        """
        Perform lung disease classification on an input image.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            Tuple of (predicted_class, confidence_score, predictions_array)
            Returns (None, None, None) if prediction fails
        """
        try:
            if self.model is None:
                if not self.load_model():
                    return None, None, None

            # Preprocess image for inference
            processed_image = preprocess_for_inference(image)

            # Perform inference - handle both named and positional inputs
            if hasattr(self.model, 'input_names') and len(self.model.input_names) > 0:
                predictions = self.model.predict({self.model.input_names[0]: processed_image}, verbose=0)
            else:
                predictions = self.model.predict(processed_image, verbose=0)

            # Handle potential list output from model
            if isinstance(predictions, list):
                predictions = predictions[0]

            # Get prediction results
            pred_idx = np.argmax(predictions[0])
            predicted_class = self.class_labels[pred_idx]
            confidence_score = float(predictions[0][pred_idx])

            return predicted_class, confidence_score, predictions[0]

        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None, None

    def get_class_labels(self) -> list:
        """
        Get the list of class labels.

        Returns:
            List of class label strings
        """
        return self.class_labels.copy()

def predict_lung_disease(image: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[np.ndarray]]:
    """
    Convenience function for single prediction.

    Args:
        image: Input image as numpy array (RGB format)

    Returns:
        Tuple of (predicted_class, confidence_score, predictions_array)
    """
    classifier = LungDiseaseClassifier()
    return classifier.predict(image)
