from os.path import dirname
import labels_model
import pandas as pd

import pytest
from datascience_model_commons.pandas import read_data

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

from labels_model.config.settings import TARGET_COLUMN


@pytest.fixture(scope="module")
def df():
    df = read_data(file_path="/tmp/labels_model_test/preprocessed_training_data")
    return df


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
                "sample_start_date": "2018-02-01",
                "sample_end_date": "2018-03-01",
            },
        ),
        training=YDSTrainingConfig(
            training_type=TrainingType.TENSORFLOW,
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
                "preprocessing": {
                    "job_type": "spark",
                    "output_path": f"{dirname(dirname(labels_model.__file__))}/test/resources/data/",
                    "docker_output_path": "/tmp/labels_model_test",
                },
                "job_type": "tensorflow",
                "docker_output_path": "/tmp/labels_model_test/",
                "docker_performant_model_output_path": "/tmp/labels_model_test/model",
            },
        ),
    )


@pytest.fixture(scope="module")
def test_mode_data() -> pd.DataFrame:
    """
    Generate test sample for test description

    :return: pandas dataframe with input columns for the model and target category
    """
    test_sample = (
        pd.DataFrame(
            columns=[
                "description",
                "amount",
                "transaction_type",
                "internal_transaction",
                "country_code",
                TARGET_COLUMN,
                "bank_counterparty_name",
                "bank_counterparty_iban",
            ]
        )
        .append(
            [
                {
                    "description": "test_description:5cb92fb6-93c5-452f-aa27-53a485b8f370:Test_music_streaming, Test_gym",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Test_music_streaming", "Test_gym"},
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "Sales_tax, Other_income",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
    )

    return test_sample


@pytest.fixture(scope="module")
def tax_test_data() -> pd.DataFrame:
    """
    Generate test sample for cucumber tests; passing all test is required to make model performant

    :return: pandas dataframe with input columns for the model and target category
    """
    test_sample = (
        pd.DataFrame(
            columns=[
                "description",
                "amount",
                "transaction_type",
                "internal_transaction",
                "country_code",
                TARGET_COLUMN,
                "bank_counterparty_name",
                "bank_counterparty_iban",
            ]
        )
        .append(
            [
                {
                    "description": "hmrc corporation t 4106400364a00102a BBP",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Tax_gb_corporate_tax"},
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "hmrc corporation t 4106400364a00102a BBP",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    TARGET_COLUMN: {"Tax_gb_corporate_tax"},
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "dvla vehicle",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Tax_gb_vehicle_tax"},
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "purple cats ate cute!",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "DE",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "teruggaaf",
                    "bank_counterparty_name": "belastingdienst",
                    "amount": 200.0,
                    "transaction_type": "credit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_iban": "NL58INGB0649306597",
                    "country_code": "NL",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "TERUGGAAF NR. 851494523V760112 VPB.2017 (JACHTTRUST )",
                    "amount": 2772.0,
                    "transaction_type": "credit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    TARGET_COLUMN: {"Tax_nl_venootschapsbelasting"},
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "TERUGGAAF NR. 851494523V760112 VPB.2017 (JACHTTRUST )",
                    "amount": 2772.0,
                    "transaction_type": "credit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    "country_code": "GB",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "Naam: BELASTINGDIENST\nOmschrijving: VJ-104-X 01-10-2020\nt/m 30-10-2020 MEER INFO\nWWW.BELASTINGDIENST.NL\nIBAN: NL86INGB0002445588\nKenmerk: COAXX579982000202010211301038114647\nMachtiging ID: 012005388\nIncassant ID: NL35ZZZ273653230000\nDoorlopende incasso\nValutadatum: 27-10-2020",
                    "amount": 28.0,
                    "transaction_type": "debit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    TARGET_COLUMN: {"Tax_nl_motorrijtuigenbelasting"},
                }
            ],
            ignore_index=True,
        )
        .append(
            [
                {
                    "description": "dvla vehicle Naam: BELASTINGDIENST\nOmschrijving: VJ-104-X 01-10-2020\nt/m 30-10-2020 MEER INFO\nWWW.BELASTINGDIENST.NL\nIBAN: NL86INGB0002445588\nKenmerk: COAXX579982000202010211301038114647\nMachtiging ID: 012005388\nIncassant ID: NL35ZZZ273653230000\nDoorlopende incasso\nValutadatum: 27-10-2020",
                    "amount": 28.0,
                    "transaction_type": "debit",
                    "account_type": "CURRENT_ACCOUNT",
                    "bank_counterparty_name": "Belastingsdienst",
                    "bank_counterparty_iban": "NL86INGB0002445588",
                    TARGET_COLUMN: {
                        "Tax_nl_motorrijtuigenbelasting",
                        "Tax_gb_vehicle_tax",
                    },
                }
            ],
            ignore_index=True,
        )
        # This sample is used in tests with build labels-serving and should succeed
        .append(
            [
                {
                    "description": "Refund",
                    "transaction_type": "debit",
                    "amount": 10.0,
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
    )
    return test_sample
