import os
from pathlib import Path
import pytest

from labels_model.preprocessing.main import preprocess
from labels_model.preprocessing.settings import (
    PREPROCESSING_METADATA_FILE,
)


@pytest.mark.first
def test_preprocess(project_config, spark_session):
    execution_date = "2020-12-22-12:22"

    preprocess(
        env="local",
        execution_date=execution_date,
        spark=spark_session,
        project_config=project_config,
    )

    script_config = project_config.preprocessing.script_config

    assert os.path.exists(
        Path(script_config["docker_output_path"]) / script_config["training_data_file"]
    )
    assert os.path.exists(
        Path(script_config["docker_output_path"])
        / script_config["validation_data_file"]
    )
    assert os.path.exists(
        Path(script_config["docker_output_path"]) / script_config["test_data_file"]
    )
    assert os.path.exists(
        Path(script_config["docker_output_path"]) / PREPROCESSING_METADATA_FILE
    )
