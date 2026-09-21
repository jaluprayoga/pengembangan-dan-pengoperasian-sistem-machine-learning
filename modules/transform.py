"""Transform module for TFX pipeline preprocessing.

This module standardizes numerical features using Z-score scaling and casts
the binary target label for diabetes risk prediction.
"""

import tensorflow as tf
import tensorflow_transform as tft

NUMERICAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
LABEL_KEY = "Outcome"


def transformed_name(key):
    """Renaming transformed features with _xf suffix."""
    return key + "_xf"


def preprocessing_fn(inputs):
    """tf.transform callback function for preprocessing input features.

    Args:
        inputs: Mapping from feature keys to raw feature tensors.

    Returns:
        Mapping from transformed feature keys to transformed tensors.
    """
    outputs = {}

    # Scale numerical features using Z-score normalization
    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_z_score(inputs[feature])

    # Pass through the label and cast to int64
    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
