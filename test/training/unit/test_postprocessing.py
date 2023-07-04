import pytest
import numpy as np
import pandas as pd
import tensorflow as tf

from labels_model.training.postprocessing import (
    BusinessRulesTransformer,
    EliminateLabelDependencies,
    PredictionsIndicatorAboveThreshold,
)

from labels_model.training.settings import (
    LABELS,
    N_TRAINING_LABELS,
    N_TRANSACTION_LABELS,
    GENERAL_SIMILARITY_THRESHOLD,
)


class PostProcessingTest(tf.test.TestCase):
    def init(self, **kwargs):
        super().__init__(**kwargs)

    @pytest.mark.first
    def test__postprocessing_layer(self):
        """Test the results after model postprocessing"""

        # extract indexes for categories that we modify in postprocessing layer
        salary_index, refund_index, bonus_money_index = (
            list(LABELS).index(label) for label in ["Salary", "Refund", "Bonus_money"]
        )

        # define input features for examples
        df = pd.DataFrame(
            [
                {
                    "scaled_amount": 100.0,
                    "is_debit_transaction": 1.0,
                    "is_internal_transaction": 0.0,
                },
                # row nr 1: debit transaction - salary False ,refund False
                {
                    "scaled_amount": 100.0,
                    "is_debit_transaction": 1.0,
                    "is_internal_transaction": 0.0,
                },
                # row nr 2: debit transaction - salary False,refund False
                {
                    "scaled_amount": 100,
                    "is_debit_transaction": 0.0,
                    "is_internal_transaction": 0.0,
                },
                # row nr 3: credit/internal transaction - salary True,refund True
                {
                    "scaled_amount": 100,
                    "is_debit_transaction": 0.0,
                    "is_internal_transaction": 1.0,
                },
                # row nr 4: credit/internal transaction - salary True, refund False
                {
                    "scaled_amount": 100,
                    "is_debit_transaction": 0.0,
                    "is_internal_transaction": 1.0,
                },
                # row nr 5: credit/internal transaction - salary True, refund False, bonus money False
                {
                    "scaled_amount": 100,
                    "is_debit_transaction": 1.0,
                    "is_internal_transaction": 0.0,
                },
                # row nr 5: credit/internal transaction - salary True, refund False, bonus money False
            ]
        )

        input_shape = len(df)

        # define random similarities below threshold
        similarities = np.random.uniform(
            low=0,
            high=GENERAL_SIMILARITY_THRESHOLD,
            size=(input_shape, N_TRAINING_LABELS),
        )

        # row nr 1: internal transaction - set max similarity for Savings above threshold
        similarities[0, salary_index] = 1.0
        similarities[1, refund_index] = 1.0
        similarities[2, refund_index] = 0.7
        similarities[2, salary_index] = 1.0
        similarities[3, salary_index] = 1.0
        similarities[4, bonus_money_index] = 1.0
        similarities[5, bonus_money_index] = 1.0
        # row nr 2: internal transaction - set max similarity for Savings below threshold
        expected_similarities = similarities.copy()
        expected_similarities[:, :] = 0.0
        expected_similarities[2, salary_index] = 1.0
        expected_similarities[3, salary_index] = 1.0
        expected_similarities = expected_similarities[:, 30:]

        # directly input similarities to postprocessing model
        input_similarities = tf.keras.layers.Input(shape=(N_TRAINING_LABELS,))

        input_is_internal_transaction = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.float32, name="internal_transaction"
        )
        input_is_debit_transaction = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.float32, name="transaction_type"
        )

        input_dict = {
            "similarities": input_similarities,
            "is_debit_transaction": input_is_debit_transaction,
            "is_internal_transaction": input_is_internal_transaction,
        }

        postprocessed_similarities = BusinessRulesTransformer(
            name="postprocessed_similarities_business_rules"
        )(input_dict)

        postprocessed_similarities = EliminateLabelDependencies(
            name="postprocessed_similarities_label_dependencies"
        )(postprocessed_similarities)

        predictions_indicator_above_threshold = PredictionsIndicatorAboveThreshold(
            name="predictions_indicator_above_threshold"
        )(postprocessed_similarities)

        postprocessed_model = tf.keras.Model(
            inputs=[
                input_is_internal_transaction,
                input_is_debit_transaction,
                input_similarities,
            ],
            outputs=predictions_indicator_above_threshold,
        )

        postprocessed_similarities = postprocessed_model.predict(
            [
                df["is_internal_transaction"],
                df["is_debit_transaction"],
                similarities,
            ]
        )

        # check the shape of predictions
        assert postprocessed_similarities.shape == (input_shape, N_TRANSACTION_LABELS)
        self.assertAllEqual(postprocessed_similarities, expected_similarities)
