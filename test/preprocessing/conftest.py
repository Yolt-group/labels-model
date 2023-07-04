from os.path import dirname
import labels_model

import pytest
from pyspark.sql import SparkSession
import datetime as dt
from datascience_model_commons.spark import get_spark_session

from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSDomain,
    YDSEnvironment,
    YDSPreprocessingConfig,
    PreprocessingType,
    YDSTrainingConfig,
    TrainingType,
    DeployingUser,
)

# from labels_model.config.local import CONFIG
from labels_model.preprocessing.data import (
    read_data_and_select_columns,
    create_training_data_frame,
)


@pytest.fixture(scope="session")
def spark_session(request) -> SparkSession:
    spark = get_spark_session("labels_test")
    request.addfinalizer(lambda: spark.stop())

    return spark


@pytest.fixture(scope="module")
def df(spark_session, project_config):
    script_config = project_config.preprocessing.script_config

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
            table=table, spark=spark_session, project_config=project_config
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

    train, _, _ = create_training_data_frame(
        transactions=transactions,
        accounts=accounts,
        users=users,
        test_users=test_users,
        user_single_feedback_created_categories=user_single_feedback_created_categories,
        user_multiple_feedback_created_categories=user_multiple_feedback_created_categories,
        user_multiple_feedback_applied_categories=user_multiple_feedback_applied_categories,
        historical_categories_feedback=historical_categories_feedback,
        historical_labels_feedback=historical_labels_feedback,
        start_training_date=dt.datetime.strptime(
            script_config.get("sample_start_date"),
            "%Y-%m-%d",
        ),
        nl_start_training_date=dt.datetime.strptime(
            script_config.get("nl_sample_start_date"),
            "%Y-%m-%d",
        ),
        end_training_date=dt.datetime.strptime(
            script_config.get("sample_end_date"),
            "%Y-%m-%d",
        ),
        n_labels_feedback=script_config.get("n_labels_feedback_per_country_per_label"),
        n_categories_feedback=script_config.get("n_categories_feedback"),
        n_validation_samples=script_config.get("n_validation_samples"),
        n_test_samples=script_config.get("n_test_samples"),
    )

    return train


@pytest.fixture(scope="module")
def user_tables(spark_session, project_config):
    (users, test_users,) = (
        read_data_and_select_columns(
            table=table, spark=spark_session, project_config=project_config
        )
        for table in [
            "users",
            "test_users",
        ]
    )
    return users, test_users


@pytest.fixture(scope="module")
def project_config() -> YDSProjectConfig:
    return YDSProjectConfig(
        model_name="labels-model",
        domain=YDSDomain.YoltApp,
        model_bucket="local",
        aws_iam_role_name="local",
        env=YDSEnvironment.DTA,
        deploy_id="local",
        deploying_user=DeployingUser(first_name="test", last_name="user"),
        git_branch="",
        git_commit_short="",
        package_dir="labels_model",
        preprocessing=YDSPreprocessingConfig(
            processing_type=PreprocessingType.SPARK,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
            script_config={
                "environment": "local",
                "model": "labels",
                "model_tag": "local-model-tag",
                "deploy_id": "local-deploy-id",
                "role": "YoltDatascienceSagemakerLabelsModel",
                "sample_start_date": "2017-01-01",
                "sample_end_date": "2020-12-31",
                "nl_sample_start_date": "2017-01-01",
                "n_labels_feedback_per_country_per_label": 10,
                "n_categories_feedback": 300,
                "n_validation_samples": 50,
                "n_test_samples": 50,
                "training_data_file": "preprocessed_training_data",
                "validation_data_file": "preprocessed_validation_data",
                "test_data_file": "preprocessed_test_data",
                "run_id": "local-run-id",
                "spark_log_level": "WARN",
                "data_file_paths": {
                    "transactions": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/transactions.csv",
                    "transaction_cycles": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/transaction_cycles.csv",
                    "accounts": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/account.csv",
                    "users": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/user.csv",
                    "test_users": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/test_users.csv",
                    "user_single_feedback_created_categories": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/data_science_events/user_single_feedback_created_categories.json",
                    "user_multiple_feedback_created_categories": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/data_science_events/user_multiple_feedback_created_categories.json",
                    "user_multiple_feedback_applied_categories": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/data_science_events/user_multiple_feedback_applied_categories.json",
                    "user_feedback_labels": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/data_science_events/user_feedback_labels.csv",
                    "historical_categories_feedback": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/historical_categories_feedback/*.csv",
                    "historical_labels_feedback": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/historical_labels_feedback/*.csv",
                },
                "job_type": "spark",
                "output_path": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/",
                "docker_output_path": "/tmp/labels_model_test",
                "training": {
                    "job_type": "tensorflow",
                    "docker_output_path": "/tmp/labels_model_test/",
                    "docker_performant_model_output_path": "/tmp/labels_model_test/model",
                },
            },
        ),
        training=YDSTrainingConfig(
            training_type=TrainingType.TENSORFLOW,
            entrypoint="main.py",
            sagemaker_processor_kwargs={},
        ),
    )
