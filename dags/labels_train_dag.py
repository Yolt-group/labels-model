import os
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime

deploy_id = "{deploy_id}"  # to be filled in during deployment
vault_context = "{vault_context}"  # to be filled in during deployment

default_args = {"provide_context": True, "start_date": datetime(2021, 7, 1)}

if os.environ["ENV"] == "management-prd":
    nexus_host = "nexus.yolt.io:443"
    nexus_address = f"https://{nexus_host}"
    extra_flags = ["--trusted-host", nexus_host]
else:
    nexus_address = "https://nexus.yolt.io:443"
    extra_flags = []

virtualenv_requirements = [
    "--extra-index-url",
    f"{nexus_address}/repository/pypi-hosted/simple",
    *extra_flags,
    "datascience_model_commons==0.3.11.3",
]


@dag(
    default_args=default_args,
    schedule_interval="0 12 * * 0",
    tags=["datascience"],
    catchup=False,
)
def labels_train():
    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
    )
    def preprocessing():
        from datascience_model_commons.airflow import airflow_run_spark_preprocessing

        airflow_run_spark_preprocessing("./dags/labels_model_yds.yaml")

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
        multiple_outputs=True,
    )
    def training():
        from datetime import datetime
        from datascience_model_commons.airflow import (
            airflow_run_tensorflow_training_job,
        )

        training_start = datetime.now()
        estimator = airflow_run_tensorflow_training_job("./dags/labels_model_yds.yaml")

        return {
            "model_artifact_uri": estimator.model_data,
            "training_run_start": training_start.strftime("%Y-%m-%d-%H-%M"),
        }

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
    )
    def copy_trained_model(trained_model_details):
        from datascience_model_commons.deploy.config.load import (
            load_config_while_in_job,
        )

        from datascience_model_commons.airflow import invoke_copy_lambda
        from pathlib import Path
        from datascience_model_commons.utils import get_logger

        logger = get_logger()
        logger.info(
            f"Going to copy trained model based on details: {trained_model_details}"
        )
        project_config = load_config_while_in_job(Path("./dags/labels_model_yds.yaml"))

        model_artifact_uri = (
            trained_model_details["model_artifact_uri"].replace("s3://", "").split("/")
        )
        destination_bucket = f"yolt-dp-{project_config.env.value}-exchange-yoltapp"
        destination_prefix = f"artifacts/datascience/{project_config.model_name}/{project_config.git_branch}/{trained_model_details['training_run_start']}"  # noqa
        destination_filename = model_artifact_uri[-1]
        invoke_copy_lambda(
            source_bucket=model_artifact_uri[0],
            source_key="/".join(model_artifact_uri[1:]),
            dst_bucket=destination_bucket,
            # This is formatted this way because of backwards compatibility.
            # Ideally, we would indicate the model artifact via a {branch, deploy_id, training_start}
            # identifier.
            dst_prefix=destination_prefix,  # noqa
            new_key=destination_filename,
        )

        return f"s3://{destination_bucket}/{destination_prefix}/{destination_filename}"

    @task.virtualenv(
        use_dill=True,
        system_site_packages=True,
        requirements=virtualenv_requirements,
    )
    def send_success_notification():
        from datascience_model_commons.airflow import (
            send_dag_finished_to_slack_mle_team,
        )

        send_dag_finished_to_slack_mle_team()

    trained = training()
    preprocessing() >> trained
    labels_copy_trained = copy_trained_model(trained)

    env = os.environ["ENV"]
    task_name = "trigger_build_labels_serving"
    if env == "management-dta":
        (
            labels_copy_trained
            >> DummyOperator(task_id=task_name)
            >> send_success_notification()
        )
    elif env == "management-prd":
        gitlab_token = Variable.get("gitlab-labels")
        payload = {
            "token": gitlab_token,
            "ref": "master",
            "variables[MODEL_URI]": labels_copy_trained,
        }

        (
            SimpleHttpOperator(
                task_id=task_name,
                http_conn_id="gitlab",
                endpoint="api/v4/projects/944/trigger/pipeline",
                method="POST",
                data=payload,
                log_response=True,
                retries=25,
            )
            >> send_success_notification()
        )


labels_train_dag = labels_train()
