import datetime as dt
import argparse
import yaml
from datascience_model_commons.deploy.config.domain import YDSProjectConfig
from datascience_model_commons.deploy.config.load import load_config_while_in_job
from datascience_model_commons.deploy.config.schema import YDSProjectConfigSchema
from datascience_model_commons.spark import get_spark_session, upload_parquet
from datascience_model_commons.utils import get_logger
from datascience_model_commons.general import upload_metadata

from labels_model.preprocessing.settings import PREPROCESSING_METADATA_FILE

import time
from pathlib import Path
from pyspark.sql import SparkSession
from typing import AnyStr

logger = get_logger()


def preprocess(
    env: AnyStr,
    execution_date: AnyStr,
    spark: SparkSession,
    project_config: YDSProjectConfig,
    output_base_path: str = "",
):
    from labels_model.preprocessing.data import (
        read_data_and_select_columns,
        create_training_data_frame,
    )

    """processing data for categories model"""

    script_config = project_config.preprocessing.script_config

    project_config_as_dict = YDSProjectConfigSchema.instance_as_dict(project_config)
    logger.info(f"Preprocessing project config: \n{project_config_as_dict}")

    # store configuration in metadata
    #
    # preprocessing_metadata goes to model.tar.gz
    #
    preprocessing_metadata = {"config": project_config_as_dict}

    start_training_date = dt.datetime.strptime(
        script_config["sample_start_date"],
        "%Y-%m-%d",
    )
    end_training_date = dt.datetime.strptime(
        script_config["sample_end_date"],
        "%Y-%m-%d",
    )

    nl_start_training_date = dt.datetime.strptime(
        script_config["nl_sample_start_date"],
        "%Y-%m-%d",
    )

    n_labels_feedback_per_country_per_label = int(
        script_config["n_labels_feedback_per_country_per_label"]
    )

    n_categories_feedback = int(script_config["n_categories_feedback"])

    n_validation_samples = int(script_config["n_validation_samples"])

    n_test_samples = int(script_config["n_test_samples"])

    logger.info("Loading data: \n")
    (
        transactions,
        accounts,
        users,
        test_users,
        user_single_feedback_created_categories,
        user_multiple_feedback_created_categories,
        user_multiple_feedback_applied_categories,
        historical_categories_feedback,
        historical_labels_feedback,
    ) = (
        read_data_and_select_columns(
            table=table, spark=spark, project_config=project_config
        )
        for table in [
            "transactions",
            "accounts",
            "users",
            "test_users",
            "user_single_feedback_created_categories",
            "user_multiple_feedback_created_categories",
            "user_multiple_feedback_applied_categories",
            "historical_categories_feedback",
            "historical_labels_feedback",
        ]
    )

    # create training data
    start_time = time.time()

    train, validation, test = create_training_data_frame(
        transactions=transactions,
        accounts=accounts,
        users=users,
        test_users=test_users,
        user_single_feedback_created_categories=user_single_feedback_created_categories,
        user_multiple_feedback_created_categories=user_multiple_feedback_created_categories,
        user_multiple_feedback_applied_categories=user_multiple_feedback_applied_categories,
        historical_categories_feedback=historical_categories_feedback,
        historical_labels_feedback=historical_labels_feedback,
        start_training_date=start_training_date,
        nl_start_training_date=nl_start_training_date,
        end_training_date=end_training_date,
        n_labels_feedback=n_labels_feedback_per_country_per_label,
        n_categories_feedback=n_categories_feedback,
        n_validation_samples=n_validation_samples,
        n_test_samples=n_test_samples,
    )
    preprocessing_time = time.time() - start_time
    preprocessing_metadata["preprocessing_time"] = time.strftime(
        "%H:%M:%S", time.gmtime(preprocessing_time)
    )

    preprocessing_metadata["training_data_size_per_country"] = training_data_size = (
        train["country_code"].value_counts().sort_index().to_dict()
    )

    preprocessing_metadata[
        "validation_data_size_per_country"
    ] = validation_data_size = (
        validation["country_code"].value_counts().sort_index().to_dict()
    )

    preprocessing_metadata["test_data_size_per_country"] = test_data_size = (
        test["country_code"].value_counts().sort_index().to_dict()
    )

    logger.info(
        f"Training data created; Number of transactions used per country: {training_data_size}"
    )

    logger.info(
        f"Validation data created; Number of transactions used per country: {validation_data_size}"
    )

    logger.info(
        f"Test data created; Number of transactions used per country: {test_data_size}"
    )

    # serialize training log
    preprocessing_metadata_yml = yaml.dump(preprocessing_metadata)
    logger.info(f"Preprocessing metadata: \n{preprocessing_metadata_yml}")

    # store datasets in parquet files
    upload_parquet(
        df=train,
        path=Path(script_config["docker_output_path"]),
        file_name=script_config["training_data_file"],
    )

    upload_parquet(
        df=validation,
        path=Path(script_config["docker_output_path"]),
        file_name=script_config["validation_data_file"],
    )

    upload_parquet(
        df=test,
        path=Path(script_config["docker_output_path"]),
        file_name=script_config["test_data_file"],
    )

    # store preprocessing metadata
    upload_metadata(
        metadata=preprocessing_metadata,
        path=Path(script_config["docker_output_path"]),
        file_name=PREPROCESSING_METADATA_FILE,
    )


if __name__ == "__main__":
    logger.info("STARTING JOB")
    parser = argparse.ArgumentParser()
    # Positional args that are provided when starting the job
    parser.add_argument("env", type=str)
    parser.add_argument("yds_config_path", type=str)
    parser.add_argument("stage", type=str)
    args, _ = parser.parse_known_args()
    project_config = load_config_while_in_job(Path(args.yds_config_path))

    app_name = f"{project_config.model_name}_preprocessing"
    spark = get_spark_session(app_name, log_level="WARN")

    preprocess(
        project_config.env.value,
        dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        spark,
        project_config,
    )

    logger.info("Finished.")
