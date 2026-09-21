"""Tuner module for TFX pipeline hyperparameter optimization.

This module searches for optimal hidden layer units and learning rate
using KerasTuner to maximize validation recall for diabetes detection.
"""

from typing import Any, Dict, NamedTuple, Text
import keras_tuner as kt
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

try:
    from modules.trainer import input_fn, model_builder
except ImportError:
    from trainer import input_fn, model_builder

# Define TunerFnResult
TunerFnResult = NamedTuple(
    "TunerFnResult",
    [
        ("tuner", kt.Tuner),
        ("fit_kwargs", Dict[Text, Any]),
    ],
)


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Build the tuner using the KerasTuner API.

    Args:
        fn_args (FnArgs): Arguments passed by TFX Tuner component.

    Returns:
        TunerFnResult: NamedTuple containing tuner and fit kwargs.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    tuner = kt.RandomSearch(
        hypermodel=model_builder,
        objective=kt.Objective("val_recall", direction="max"),
        max_trials=10,
        seed=1,
        directory=fn_args.working_dir,
        project_name="diabetes_tuning",
    )

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
            "class_weight": {0: 1.0, 1: 2.5},
        },
    )
