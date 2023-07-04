import pytest

from labels_model.config.settings import TARGET_COLUMN
from labels_model.training.main import train


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


@pytest.mark.second
def test_train(project_config):
    script_config = project_config.training.script_config
    preprocess_path = script_config["preprocessing"]["docker_output_path"]
    train(project_config=project_config, preprocess_path=preprocess_path)
    assert True


def run_model_with_data(project_config, df, test_label_prefix):
    script_config = project_config.training.script_config
    model_path = script_config["docker_output_path"]

    # Load model to test custom model code
    from labels_model.predict.model import Model

    model = Model(model_path)

    # Predict labels
    model_predictions = df.assign(labels=model.predict(df=df))

    # Filter on test label prefix
    model_predictions["labels"] = model_predictions["labels"].apply(
        lambda labels: {label for label in labels if test_label_prefix in label}
    )

    for i in range(0, len(model_predictions)):
        print(
            f"Cucumber test no {i+1}: Target = {model_predictions.loc[i, 'target_label']}, Prediction = {model_predictions.loc[i, 'labels']}"
        )

        if (
            model_predictions.loc[i, TARGET_COLUMN]
            != model_predictions.loc[i, "labels"]
        ):
            assert False

    assert True


@pytest.mark.last
def test_labels_tax_labels_in_model(project_config, tax_test_data):
    run_model_with_data(project_config, tax_test_data, test_label_prefix="Tax_")


@pytest.mark.last
def test_test_mode_in_model(project_config, test_mode_data):
    run_model_with_data(project_config, test_mode_data, test_label_prefix="Test_")
