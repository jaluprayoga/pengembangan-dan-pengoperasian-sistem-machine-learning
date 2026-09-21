"""Trainer module for TFX pipeline model training and export.

This module builds, trains, evaluates, and exports the Diabetes Prediction
Deep Neural Network model with serving signatures.
"""

import keras_tuner as kt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

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


def transformed_name(key: str) -> str:
    """Generate transformed feature name with _xf suffix."""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed TFRecord data."""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs=None, batch_size=32):
    """Generates features and label for tuning/training."""
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


def model_builder(hp):
    """Build machine learning model with Hyperparameter tuning."""
    inputs = {}
    for feature in NUMERICAL_FEATURES:
        inputs[transformed_name(feature)] = tf.keras.Input(
            shape=(1,),
            name=transformed_name(feature),
            dtype=tf.float32,
        )

    # Concatenate all inputs
    x = tf.keras.layers.Concatenate()(list(inputs.values()))

    # Hyperparameters
    hp_units_1 = hp.Int("units_1", min_value=16, max_value=128, step=16)
    hp_units_2 = hp.Int("units_2", min_value=8, max_value=64, step=8)
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3])

    x = layers.Dense(hp_units_1, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(hp_units_2, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):
    """Returns a function that parses a serialized tf.Example."""
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed_features = model.tft_layer(parsed_features)

        final_features = {
            transformed_name(f): transformed_features[transformed_name(f)]
            for f in NUMERICAL_FEATURES
        }

        return model(final_features)

    return serve_tf_examples_fn


def run_fn(fn_args: FnArgs):
    """Train the model based on given args."""
    transform_graph_path = (
        fn_args.transform_graph_path
        if hasattr(fn_args, "transform_graph_path") and fn_args.transform_graph_path
        else getattr(fn_args, "transform_output", None)
    )
    tf_transform_output = tft.TFTransformOutput(transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=None)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=None)

    if fn_args.hyperparameters:
        if isinstance(fn_args.hyperparameters, kt.HyperParameters):
            hparams = fn_args.hyperparameters
        elif isinstance(fn_args.hyperparameters, dict):
            hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
        else:
            hparams = kt.HyperParameters.from_config(fn_args.hyperparameters)
    else:
        hparams = kt.HyperParameters()
        hparams.Fixed("units_1", value=64)
        hparams.Fixed("units_2", value=32)
        hparams.Fixed("learning_rate", value=1e-3)

    model = model_builder(hparams)

    class_weight = {0: 1.0, 1: 2.5}

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True,
    )

    print("\n" + "=" * 60)
    print("[TRAINER] MODEL ARCHITECTURE & TRAINING CONFIGURATION")
    print("=" * 60)
    model.summary(print_fn=print)
    print("\nTraining Configuration:")
    print(f"  * Optimizer: Adam (learning_rate={hparams.get('learning_rate')})")
    print("  * Loss Function: binary_crossentropy")
    print(f"  * Class Weighting: {class_weight}")
    print("  * Callbacks: [EarlyStopping(monitor='val_loss', patience=5)]")
    print("=" * 60 + "\n")

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=20,
        class_weight=class_weight,
        callbacks=[early_stopping],
    )

    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }

    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
