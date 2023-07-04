import argparse
import datetime as dt
import os
import time
from pathlib import Path
import yaml

from datascience_model_commons.deploy.config.domain import (
    YDSProjectConfig,
    YDSEnvironment,
)
from datascience_model_commons.deploy.config.load import load_config_while_in_job
from datascience_model_commons.deploy.config.schema import YDSProjectConfigSchema

from datascience_model_commons.pandas import read_data
from labels_model.training.utils import copy_model_and_metadata
from labels_model.config.settings import TRANSACTION_LABELS

from datascience_model_commons.utils import get_logger

from labels_model.training.model import (
    TransactionLabelClassifier,
    create_cucumber_test_sample,
    check_performance,
    preprocess_test_set,
)
from labels_model.training.settings import (
    COUNTRIES,
)

import traceback

logger = get_logger()


def train(project_config: YDSProjectConfig, preprocess_path):
    """Training labels model"""

    # store configuration in training metadata
    training_metadata = {
        "config": YDSProjectConfigSchema.instance_as_dict(project_config),
        "labels": TRANSACTION_LABELS,
    }

    script_config = project_config.training.script_config

    # read preprocessed datasets
    (train, validation, test) = (
        read_data(file_path=table_path)
        for table_path in [
            f"{preprocess_path}/{script_config.get('training_data_file') }",
            f"{preprocess_path}/{script_config.get('validation_data_file') }",
            f"{preprocess_path}/{script_config.get('test_data_file') }",
        ]
    )

    # instantiate new estimator
    model = TransactionLabelClassifier(script_config)
    logger.info("TransactionLabelClassifier initialized")

    # fit
    start_time = time.time()
    model.fit(df_train=train, df_validation=validation)
    fit_time = time.time() - start_time
    training_metadata["fit_time"] = time.strftime("%H:%M:%S", time.gmtime(fit_time))
    logger.info("Model fitted")

    # evaluate on test set
    t = time.time()
    preprocessed_test = preprocess_test_set(df=test)

    metrics = dict()
    for country in COUNTRIES.keys():
        training_metadata[country] = metrics[country] = model.evaluate(
            df=preprocessed_test[
                preprocessed_test["country_code"].isin(COUNTRIES[country])
            ]
        )
    training_metadata["evaluate_time"] = time.time() - t
    logger.info("Model evaluated on test set")

    # Temporarily save model
    test_path = Path("/tmp") / Path(project_config.model_name)
    copy_model_and_metadata(
        docker_output_path=test_path,
        model=model,
        training_metadata=training_metadata,
    )
    # Load model to test custom model code
    from labels_model.predict.model import Model

    production_model = Model(test_path)

    # check cucumber tests & model performance
    cucumber_test_sample = create_cucumber_test_sample()
    cucumber_predicted_labels = production_model.predict(df=cucumber_test_sample)

    training_metadata["is_performant"] = is_performant = check_performance(
        metrics=metrics,
        cucumber_tests_df=cucumber_test_sample,
        predictions_df=cucumber_predicted_labels,
    )

    # serialize training log
    training_metadata_yml = yaml.dump(training_metadata)
    logger.info(f"Training metadata: \n{training_metadata_yml}")

    # always save model in docker output path as "model.tar.gz" artifact (automatically created by AWS)
    copy_model_and_metadata(
        docker_output_path=Path(script_config.get("docker_output_path")),
        model=model,
        training_metadata=training_metadata,
    )

    if is_performant:
        logger.info("Model performance meets expectations")

        # if the model is performant: store the same model as "output.tar.gz" artifact (automatically created by AWS)
        copy_model_and_metadata(
            docker_output_path=Path(
                script_config.get("docker_performant_model_output_path")
            ),
            model=model,
            training_metadata=training_metadata,
        )

    else:
        logger.warning("Model performance is below expectations")
        if (
            project_config.git_branch == "master"
            and project_config.env == YDSEnvironment.PRD
        ):
            raise Exception("The model wasn't performant")


if __name__ == "__main__":
    logger.info("STARTING JOB")
    # extract model directory in order to pass execution_date to output paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--config", type=str, default=os.environ["SM_CHANNEL_CONFIG"])
    parser.add_argument(
        "--preprocessing_output",
        type=str,
        default=os.environ["SM_CHANNEL_PREPROCESSING_OUTPUT"],
    )
    args, _ = parser.parse_known_args()
    logger.info(f"Going to load config from {args.config}")
    logger.info(f"Preprocessing output located in {args.preprocessing_output}")
    logger.info(
        f"Preprocessing output files {list(Path(args.preprocessing_output).glob('*'))}"
    )

    project_config: YDSProjectConfig = load_config_while_in_job(
        Path(args.config) / "yds.yaml"
    )

    script_config = project_config.training.script_config

    name = "training"

    output_path = args.model_dir
    execution_date = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    try:
        logger.info(
            f"{name} started at {dt.datetime.now()} airflow run_id: {execution_date}"
        )
        train(project_config=project_config, preprocess_path=args.preprocessing_output)
        logger.info(f"{name} finished at {dt.datetime.now()}")
    except Exception as e:
        trc = traceback.format_exc()
        error_string = "Exception during " + name + ": " + str(e) + "\n" + trc
        logger.error(error_string)
        # Write out error details, this will be returned as the ExitMessage in the job details
        with open("/opt/ml/output/message", "w") as s:
            s.write(error_string)
        raise e
