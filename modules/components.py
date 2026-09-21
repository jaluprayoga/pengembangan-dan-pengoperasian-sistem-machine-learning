"""Components module for initializing all TFX pipeline components.

This module sets up ExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
Transform, Tuner, Trainer, Resolver, Evaluator, and Pusher.
"""

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,no-member

from typing import List
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing


def init_components(
    data_dir: str,
    transform_module_file: str,
    tuner_module_file: str,
    trainer_module_file: str,
    serving_model_dir: str,
    use_tuner: bool = True,
) -> List[object]:
    """Initialize all components for the TFX pipeline.

    Args:
        data_dir (str): Directory containing raw input CSV dataset.
        transform_module_file (str): Path to transform Python module file.
        tuner_module_file (str): Path to tuner Python module file.
        trainer_module_file (str): Path to trainer Python module file.
        serving_model_dir (str): Export path for serving model.
        use_tuner (bool): Whether to include Tuner component in pipeline.

    Returns:
        List[object]: List of configured TFX pipeline components.
    """
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    example_gen = CsvExampleGen(input_base=data_dir, output_config=output_config)

    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])

    schema_gen = SchemaGen(
        statistics=statistics_gen.outputs["statistics"],
        infer_feature_shape=True,
    )

    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )

    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=transform_module_file,
    )

    tuner = None
    if use_tuner:
        tuner = Tuner(
            module_file=tuner_module_file,
            examples=transform.outputs["transformed_examples"],
            transform_graph=transform.outputs["transform_graph"],
            schema=schema_gen.outputs["schema"],
            train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=20),
            eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=5),
        )

    trainer_kwargs = {
        "module_file": trainer_module_file,
        "examples": transform.outputs["transformed_examples"],
        "transform_graph": transform.outputs["transform_graph"],
        "schema": schema_gen.outputs["schema"],
        "train_args": trainer_pb2.TrainArgs(splits=["train"], num_steps=20),
        "eval_args": trainer_pb2.EvalArgs(splits=["eval"], num_steps=5),
    }

    if tuner:
        trainer_kwargs["hyperparameters"] = tuner.outputs["best_hyperparameters"]

    trainer = Trainer(**trainer_kwargs)

    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")

    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="Outcome")],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.60}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -0.1},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(
                        class_name="Recall",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.70}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -0.1},
                            ),
                        ),
                    ),
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="Precision"),
                    tfma.MetricConfig(class_name="BinaryCrossentropy"),
                ]
            )
        ],
    )

    evaluator = Evaluator(
        examples=example_gen.outputs["examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )

    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = [
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
    ]

    if tuner:
        components.append(tuner)

    components.extend([
        trainer,
        model_resolver,
        evaluator,
        pusher,
    ])

    return components
