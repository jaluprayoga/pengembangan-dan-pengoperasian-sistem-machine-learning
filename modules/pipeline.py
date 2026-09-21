"""Pipeline module defining the TFX pipeline using Apache Beam DAG orchestrator.

This module initializes the TFX pipeline with metadata connection and component list.
"""

# pylint: disable=protected-access

import os
from typing import Any, Iterable, List
import dill
from google.protobuf.message import Message
from google.protobuf.pyext import _message
from tfx.orchestration.portable.mlmd import execution_lib
from tfx.orchestration import pipeline
from tfx.orchestration.metadata import sqlite_metadata_connection_config


def _apply_runtime_compatibility_patches() -> None:
    """Apply runtime compatibility patches for Apache Beam and MLMD on Windows."""
    orig_save_type = dill._dill.save_type

    def custom_save_type(pickler: Any, obj: Any, postproc_list: Any = None) -> Any:
        if isinstance(obj, type) and issubclass(obj, Message):
            return pickler.save_global(obj)
        return orig_save_type(pickler, obj, postproc_list)

    dill._dill.save_type = custom_save_type

    try:
        _message.CMessage = _message.Message
    except AttributeError:
        pass

    if os.name == "nt":
        def win_get_executions_associated_with_all_contexts(
            metadata_handler: Any,
            contexts: Iterable[Any],
        ) -> List[Any]:
            context_ids = [c.id for c in contexts]
            if not context_ids:
                return []
            execution_sets = [
                {e.id for e in metadata_handler.store.get_executions_by_context(cid)}
                for cid in context_ids
            ]
            common_ids = set.intersection(*execution_sets) if execution_sets else set()
            if not common_ids:
                return []
            return metadata_handler.store.get_executions_by_id(list(common_ids))

        execution_lib.get_executions_associated_with_all_contexts = (
            win_get_executions_associated_with_all_contexts
        )


def init_pipeline(
    pipeline_name: str,
    pipeline_root: str,
    metadata_path: str,
    components: List[object],
) -> pipeline.Pipeline:
    """Initialize TFX pipeline object.

    Args:
        pipeline_name (str): Name of the pipeline.
        pipeline_root (str): Root directory to store pipeline artifacts.
        metadata_path (str): File path for SQLite metadata store.
        components (List[object]): List of TFX pipeline components.

    Returns:
        pipeline.Pipeline: Initialized TFX Pipeline object.
    """
    _apply_runtime_compatibility_patches()

    metadata_config = sqlite_metadata_connection_config(metadata_path)

    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=metadata_config,
        components=components,
    )
