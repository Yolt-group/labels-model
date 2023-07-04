from pathlib import Path
from typing import AnyStr, Set

from datascience_model_commons.model import BaseModel, ModelException

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

DEFAULT_VALUES_DICT = {tf.string: (tf.string, ""), tf.float32: (tf.float32, 0.0)}


# Enable loading of metadata without model source code
class SafeLoaderIgnoreUnknown(yaml.SafeLoader):
    def ignore_unknown(self, node):
        return None


class Model(BaseModel):
    def __init__(self, model_path):
        model_directory = Path(__file__).parent
        with open(model_directory / "config.yaml", "r") as f:
            self.config_dict = yaml.load(f)
            self.test_description_prefix = self.config_dict.get(
                "TEST_DESCRIPTION_PREFIX"
            )

        from datascience_model_commons.transactions.business_rules.dutch_tax_rules import (
            apply_tax_rules as apply_nl_tax_rules,
        )
        from datascience_model_commons.transactions.business_rules.uk_tax_rules import (
            apply_tax_rules as apply_uk_tax_rules,
        )

        self.tax_function_dict = {
            "NL": apply_nl_tax_rules,  # function reference
            "GB": apply_uk_tax_rules,  # function reference
        }
        self.load_model(model_path=model_path)

    def load_model(self, model_path):
        def find_artifact(root):
            import os

            # Traverse root to find model signature files (.pb file extension)
            for root, dirs, files in os.walk(root):
                for file in files:
                    if file.endswith(".pb"):
                        return root

            raise ModelException("No .pb model artifact found!")

        try:
            artifact_location = find_artifact(root=model_path)
            model = tf.saved_model.load(artifact_location)
        except Exception as e:
            raise ModelException(f"Unable to load model {model_path}") from e

        metadata_path = f"{model_path}/training_metadata.yaml"
        try:
            with open(metadata_path, "r") as stream:
                SafeLoaderIgnoreUnknown.add_constructor(
                    None, SafeLoaderIgnoreUnknown.ignore_unknown
                )
                root = yaml.load(stream, Loader=SafeLoaderIgnoreUnknown)
                labels = root["labels"]
        except IOError as ioerr:
            raise ModelException(
                f'Error reading yaml config file: "{metadata_path}"'
            ) from ioerr
        except yaml.YAMLError as yerr:
            raise ModelException(f'Unable to parse "{metadata_path}"') from yerr

        self.model = model
        self.labels = np.array(labels)
        self.predict_function = self.model.signatures["serving_default"]
        self.input_tensors = [
            tensor
            for tensor in self.predict_function.inputs
            if "unknown" not in tensor.name
        ]
        self.input_description = {
            tensor.name.split(":")[0]: DEFAULT_VALUES_DICT.get(tensor.dtype)
            for tensor in self.input_tensors
        }

    def df_to_tensor_dict(self, df):
        N = df.shape[0]
        return {
            key: np.atleast_2d(
                df[key].fillna(default_value).values.astype(dtype=dtype.as_numpy_dtype)
                if key in df
                else np.repeat(default_value, N)
            ).T
            for key, (dtype, default_value) in self.input_description.items()
        }

    def _create_labels_set(self, trx_scores):
        return set(self.labels[trx_scores == 1.0])

    def scores_to_labels(self, scores):
        return np.apply_along_axis(self._create_labels_set, 1, scores)

    @staticmethod
    def append_labels(
        prefix: AnyStr,
        input_df: pd.DataFrame,
        add_label_column: AnyStr,
        label_column: AnyStr = "labels",
    ) -> pd.DataFrame:
        def map_to_label(record: pd.Series) -> Set[AnyStr]:
            current_labels = record[label_column]
            add_label = record[add_label_column]

            if type(add_label) == str:
                add_label = add_label.lower()
                add_label = f"{prefix}_{add_label}"
                current_labels.add(add_label)

            return current_labels

        input_df[label_column] = input_df.apply(map_to_label, axis=1)
        return input_df

    def add_tax_labels(self, input_df: pd.DataFrame):
        # Make sure country code is available
        if "country_code" not in input_df:
            input_df = input_df.assign(country_code="")
        else:
            input_df["country_code"].fillna("", inplace=True)

        # Add labels for all country_code tax rules
        for country_code, tax_func in self.tax_function_dict.items():
            country_code_match = input_df["country_code"] == country_code
            country_code_missing = input_df["country_code"] == ""
            subset_df = input_df.loc[country_code_match | country_code_missing]

            # Nothing to do continue to next country code
            if len(subset_df) == 0:
                continue

            # Temporarily assign country code if missing
            subset_df = tax_func(subset_df.assign(country_code=country_code))
            prefix = f"Tax_{country_code.lower()}"
            subset_df["labels"] = self.append_labels(
                prefix=prefix, input_df=subset_df, add_label_column="tax_label"
            )
        return input_df

    def apply_business_rules(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Apply business rules here, category column should be overwritten where business rules apply"""
        input_df = self.add_tax_labels(input_df=input_df)
        input_df = self.apply_test_description_rule(input_df=input_df)
        return input_df[["labels"]]

    def apply_test_description_rule(self, input_df: pd.DataFrame) -> pd.DataFrame:
        input_df.fillna("", inplace=True)
        is_test_transaction = input_df["description"].str.startswith(
            self.test_description_prefix
        )
        subset_df = input_df.loc[is_test_transaction]
        regex_pattern = "([A-Z][a-z_]+)"
        processed_labels = (
            subset_df["description"]
            .str.extractall(regex_pattern)
            .groupby(level=0)
            .agg(set)
        )
        input_df.loc[is_test_transaction, "labels"] = processed_labels.values
        return input_df

    def predict(self, *, df: pd.DataFrame):
        if df.empty:  # Nothing to predict so return empty dataframe
            scores = pd.Series(dtype=object)
        else:
            predictions = self.predict_function(**self.df_to_tensor_dict(df))
            scores = list(predictions.values())[0].numpy()

        return self.apply_business_rules(
            input_df=df.assign(
                labels=self.scores_to_labels(scores),
            )
        )
