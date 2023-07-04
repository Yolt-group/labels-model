from setuptools import setup

ds_model_commons_version = "0.3.11.3"

# Parsing "training_requirements.txt" to be included in setup.py
# "training_requirements.txt" is still required to make SageMaker training run properly
with open("training_requirements.txt") as tr:
    training_requirements = [
        x for x in tr.read().splitlines() if not x.startswith("--")
    ]

setup(
    name="labels-model",
    version="0.2.1",
    url="https://git.yolt.io/datascience/labels/labels-model.git",
    description="Labels model",
    setup_requires=["pytest-runner"],
    install_requires=[
        f"datascience_model_commons=={ds_model_commons_version}",
        "protobuf==3.20.*",
    ],
    extras_require={
        "test": ["pytest-ordering==0.6", "pytest==5.3.5"],
        "preprocessing": [
            "pyspark==3.1.1",
        ],
        "training": training_requirements,
        "dev": [
            "black==20.8b1",
            "flake8==4.0.0",
        ],
    },
    classifiers=["Programming Language :: Python :: 3.7"],
)
