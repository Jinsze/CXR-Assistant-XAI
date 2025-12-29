"""
Grad-CAM++ implementation for explainable AI in lung disease classification.
Provides visual explanations of model decisions using activation maps.
"""

import cv2
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple
import matplotlib.pyplot as plt

def get_gradcam_plus_plus(model: tf.keras.Model,
                         img_tensor: np.ndarray,
                         class_idx: int,
                         layer_name: str = "top_conv") -> np.ndarray:
    """
    Compute Grad-CAM++ heatmap for a given image and class.

    This implementation follows the Grad-CAM++ algorithm which provides
    better visual explanations compared to standard Grad-CAM by using
    second-order gradients and improved weighting.

    Args:
        model: Trained Keras model
        img_tensor: Preprocessed input image tensor (batch_size, H, W, C)
        class_idx: Index of the class to explain
        layer_name: Name of the convolutional layer to use for Grad-CAM

    Returns:
        Normalized heatmap as numpy array (H, W)
    """
    try:
        # Create gradient model that outputs both conv features and predictions
        grad_model = tf.keras.Model(
            model.inputs,
            [model.get_layer(layer_name).output, model.output]
        )

        # Compute Grad-CAM++ using second-order gradients
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                conv_outputs, predictions = grad_model(img_tensor)

                # Handle list-based model outputs
                if isinstance(predictions, list):
                    predictions = predictions[0]

                # Get prediction for target class
                loss = predictions[:, class_idx]

            # First derivative
            grads = tape2.gradient(loss, conv_outputs)

        # Second derivative
        grads2 = tape1.gradient(grads, conv_outputs)

        # Extract first sample from batch
        conv_outputs = conv_outputs[0]
        grads = grads[0]
        grads2 = grads2[0]

        # Grad-CAM++ weighting formula
        numerator = grads2
        denominator = 2 * grads2 + tf.reduce_sum(conv_outputs * grads2, axis=(0, 1)) + 1e-10
        alpha = numerator / denominator

        # Compute weights
        weights = tf.reduce_sum(alpha * tf.nn.relu(grads), axis=(0, 1))

        # Generate heatmap
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)

        # Apply ReLU and normalize
        cam = tf.nn.relu(cam).numpy()
        cam = cam / (np.max(cam) + 1e-10)  # Avoid division by zero

        return cam

    except Exception as e:
        print(f"Error computing Grad-CAM++: {str(e)}")
        # Return zero heatmap on error
        return np.zeros((img_tensor.shape[1], img_tensor.shape[2]))

def create_gradcam_overlay(original_image: np.ndarray,
                          heatmap: np.ndarray,
                          alpha: float = 0.4) -> np.ndarray:
    """
    Create overlay of original image and Grad-CAM heatmap.

    Args:
        original_image: Original image (H, W, 3)
        heatmap: Grad-CAM heatmap (H, W)
        alpha: Transparency factor for overlay (0-1)

    Returns:
        Overlay image with heatmap applied
    """
    try:
        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

        # Convert to color heatmap
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Convert to RGB for consistency
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Create overlay
        overlay = cv2.addWeighted(original_image.astype(np.uint8), 1-alpha,
                                heatmap_rgb, alpha, 0)

        return overlay

    except Exception as e:
        print(f"Error creating overlay: {str(e)}")
        return original_image

def generate_explanation_visualization(model: tf.keras.Model,
                                     preprocessed_image: np.ndarray,
                                     original_image: np.ndarray,
                                     class_idx: int,
                                     layer_name: str = "top_conv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complete explanation visualization with Grad-CAM++.

    Args:
        model: Trained Keras model
        preprocessed_image: Preprocessed input image tensor (1, H, W, C) - used for model prediction
        original_image: Original unprocessed image (H, W, C) as uint8 - used for display
        class_idx: Predicted class index
        layer_name: Convolutional layer name for Grad-CAM

    Returns:
        Tuple of (original_image, gradcam_overlay)
    """
    try:
        # Use the provided original image for display
        original_display = original_image.astype(np.uint8)

        # Generate Grad-CAM++ heatmap using preprocessed image
        heatmap = get_gradcam_plus_plus(model, preprocessed_image, class_idx, layer_name)

        # Create overlay on original image
        overlay = create_gradcam_overlay(original_display, heatmap)

        return original_display, overlay

    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        # Return original image duplicated on error
        return original_image.astype(np.uint8), original_image.astype(np.uint8)
