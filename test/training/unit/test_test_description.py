import pytest

from labels_model.config.settings import TARGET_COLUMN


@pytest.mark.last
def test_apply_test_description_rule(project_config, test_mode_data):
    script_config = project_config.training.script_config
    model_path = script_config["docker_output_path"]

    # Load model to test custom model code
    from labels_model.predict.model import Model

    test_mode_data = test_mode_data.copy()
    test_mode_data["labels"] = [set()] * len(test_mode_data)
    model = Model(model_path)
    result = model.apply_test_description_rule(input_df=test_mode_data)

    for i in range(0, len(result)):
        print(
            f"Cucumber test no {i + 1}: Target = {result.loc[i, 'target_label']}, Prediction = {result.loc[i, 'labels']}"
        )

        if result.loc[i, TARGET_COLUMN] != result.loc[i, "labels"]:
            assert False

    assert True
