import shutil
from pathlib import Path
from datascience_model_commons.general import upload_artifact, upload_metadata
from labels_model.training.settings import MODEL_ARTIFACT_FILE, MODEL_METADATA_FILE
from labels_model.predict import model as servable_model


# always save model in docker output path as "model.tar.gz" artifact (automatically created by AWS)
def copy_model_and_metadata(docker_output_path: Path, model, training_metadata):
    upload_artifact(
        model=model,
        path=docker_output_path,
        file_name=MODEL_ARTIFACT_FILE,
    )

    # Copy all predict code and dependencies
    predict_root = Path(servable_model.__file__).parent
    for origin in predict_root.rglob("*"):
        destination = docker_output_path / origin.relative_to(predict_root)
        if origin.is_file():
            shutil.copyfile(origin, destination)
        elif origin.is_dir():
            destination.mkdir(exist_ok=True)

    # store metadata
    training_metadata.update(model.metadata)
    upload_metadata(
        metadata=training_metadata,
        path=docker_output_path,
        file_name=MODEL_METADATA_FILE,
    )
