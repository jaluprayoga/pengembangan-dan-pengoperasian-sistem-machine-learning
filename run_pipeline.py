"""Pipeline runner script to execute the TFX pipeline via Apache Beam orchestrator."""

import os
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import init_components
from modules.pipeline import init_pipeline

PIPELINE_NAME = "jaluprayoga-pipeline"
SCHEMA_PIPELINE_NAME = "jaluprayoga-tfx-pipeline"
PIPELINE_ROOT = os.path.join(PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")
SERVING_MODEL_DIR = os.path.join("serving_model")
DATA_DIR = "data"
TRANSFORM_MODULE_FILE = "modules/transform.py"
TUNER_MODULE_FILE = "modules/tuner.py"
TRAINER_MODULE_FILE = "modules/trainer.py"


def run() -> None:
    """Initialize components and execute TFX pipeline using BeamDagRunner."""
    print("=" * 60)
    print(f"INITIALIZING TFX PIPELINE: {PIPELINE_NAME}")
    print("=" * 60)

    print("\n[1/3] Setting up pipeline components...")
    components = init_components(
        data_dir=DATA_DIR,
        transform_module_file=TRANSFORM_MODULE_FILE,
        tuner_module_file=TUNER_MODULE_FILE,
        trainer_module_file=TRAINER_MODULE_FILE,
        serving_model_dir=SERVING_MODEL_DIR,
        use_tuner=True,
    )
    print(f"Successfully initialized {len(components)} components.")

    print("\n[2/3] Initializing pipeline DAG...")
    beam_pipeline = init_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        metadata_path=METADATA_PATH,
        components=components,
    )

    print("\n[3/3] Running pipeline with BeamDagRunner...")
    BeamDagRunner().run(pipeline=beam_pipeline)

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Model exported to: {SERVING_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run()
