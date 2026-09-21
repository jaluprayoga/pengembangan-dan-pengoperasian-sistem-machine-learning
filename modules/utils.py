"""Utility module for dataset loading and shared pipeline helper functions."""

from typing import Optional
import tensorflow as tf
import tensorflow_transform as tft
from modules.transform import LABEL_KEY, transformed_name


def gzip_reader_fn(filenames: tf.Tensor) -> tf.data.TFRecordDataset:
    """Load compressed TFRecord dataset files.

    Args:
        filenames (tf.Tensor): File patterns or list of filenames.

    Returns:
        tf.data.TFRecordDataset: GZIP-decompressed TFRecord dataset.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern: str,
    tf_transform_output: tft.TFTransformOutput,
    num_epochs: Optional[int] = None,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Generate batched dataset containing features and label.

    Args:
        file_pattern (str): File path pattern for input TFRecord files.
        tf_transform_output (tft.TFTransformOutput): Transform output artifact.
        num_epochs (Optional[int]): Number of passes through data (None for infinite).
        batch_size (int): Size of batches returned.

    Returns:
        tf.data.Dataset: Batched and mapped TensorFlow Dataset.
    """
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset
