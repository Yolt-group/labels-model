import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from labels_model.training.model import (
    calculate_description_length_distribution,
    generate_preprocessing_parameters,
    preprocessing_fn,
    ZeroMaskedAverage,
    dummy_loss,
    LabelEmbeddingSimilarity,
    compute_model_metrics,
    check_performance,
)
from labels_model.training.settings import (
    SEQUENCE_LENGTH,
    N_NUMERIC_COLUMNS,
    EMBEDDING_SIZE,
    N_TRAINING_LABELS,
    TRANSACTION_LABELS,
    N_TRANSACTION_LABELS,
    TARGET_COLUMN,
    TARGET_COLUMN_INT,
)


def test_calculate_description_length_distribution(df):
    input_description_length_distribution = calculate_description_length_distribution(
        df
    )

    # check last element (first element might not be stable due to some randomization inside the model)
    assert input_description_length_distribution[250] == 0.0

    # check if histogram is properly normalized
    np.testing.assert_almost_equal(
        sum(input_description_length_distribution.values()), 1, decimal=1
    )


def test_preprocessing_fn():
    """Test the data preprocessing"""

    # define pd input example to generate preprocessing params
    df = pd.DataFrame.from_dict(
        {
            "description": ["Avro Energy Energy 5"],
            "amount": [4000.0],
            "internal_transaction": ["12341id"],
            "transaction_type": ["debit"],
        }
    )

    # define tf dataset to test preprocessing function
    inputs = {
        "description": tf.constant(df["description"], shape=[]),
        "amount": tf.constant(np.float32(df["amount"]), shape=[]),
        "internal_transaction": tf.constant(df["internal_transaction"], shape=[]),
        "transaction_type": tf.constant(df["transaction_type"], shape=[]),
    }

    # FIXME: move tensorflow session to fixtures using tf.test

    generate_preprocessing_parameters(df=df)
    outputs = preprocessing_fn(inputs=inputs)

    # check the output shape of preprocessed text input
    assert outputs["preprocessed_description"].shape == (SEQUENCE_LENGTH,)

    # check the output shape of preprocessed numeric input
    assert outputs["preprocessed_numeric_features"].shape == (N_NUMERIC_COLUMNS,)

    # check whether five elements in preprocessed text are non zero - since we create 1-gram and 2-grams on words
    assert np.count_nonzero(outputs["preprocessed_description"]) == 7

    # check whether number of unique elements in preprocessed text is equal to the number of expected ngrams
    assert len(np.unique(outputs["preprocessed_description"])) == 7


@pytest.mark.first
def test_description_embedding_masking():
    """Test masking logic while generating sentence embedding"""

    # define the same architecture logic as in model.py
    input_text = tf.keras.layers.Input(shape=(4,), dtype=tf.int32)
    x = input_text
    x = tf.keras.layers.Embedding(
        input_dim=50, input_length=4, output_dim=8, mask_zero=True
    )(x)
    x = ZeroMaskedAverage()(x)
    embedding = x

    # train dummy model
    model = tf.keras.Model(inputs=input_text, outputs=embedding)
    model.compile(optimizer="adam", loss=dummy_loss)
    model.fit(np.array([[32, 22, 0, 0]]), np.array([0]), epochs=0, verbose=0)

    # generate predictions
    prediction_for_zeros = model.predict(np.array([[0, 0, 0, 0]]))
    prediction_for_known_word = model.predict(np.array([[32, 0, 0, 0]]))
    prediction_for_known_word_occurred_twice = model.predict(np.array([[32, 32, 0, 0]]))

    # embedding for padded index should be filled in with zeros
    assert (prediction_for_zeros == 0).all()

    # embedding for a sentence with known word should not be zeros
    assert (prediction_for_known_word != 0).any()

    # embedding for a sentence with two same words should be the same as with one word - since we take the mean
    assert (prediction_for_known_word == prediction_for_known_word_occurred_twice).all()


@pytest.mark.first
def test_label_embedding_similarity():
    """Test custom layer: whether the shape is as expected"""
    # define random inputs
    np.random.seed(0)
    transaction_embeddings = np.random.rand(1, EMBEDDING_SIZE)

    # define model architecture
    input_transaction_embeddings = tf.keras.layers.Input(shape=(EMBEDDING_SIZE,))
    similarities = LabelEmbeddingSimilarity()(input_transaction_embeddings)
    model = tf.keras.Model(inputs=input_transaction_embeddings, outputs=similarities)

    predicted_similarities = model.predict(transaction_embeddings)

    # check if the output shape is as expected
    assert predicted_similarities.shape == (1, N_TRAINING_LABELS)


@pytest.mark.first
def test_compute_model_metrics():
    """Test whether the classification & ranking metrics are calculated correctly"""
    # create test data with target columns used to compute metrics
    df = pd.DataFrame().assign(
        **{
            TARGET_COLUMN: TRANSACTION_LABELS,
            TARGET_COLUMN_INT: np.arange(0, N_TRANSACTION_LABELS),
        }
    )

    # add dummy predictions
    y_score = np.tile(
        np.linspace(start=0, stop=1, num=N_TRANSACTION_LABELS), (len(df), 1)
    )
    y_pred = np.array(TRANSACTION_LABELS)[np.argmax(y_score, axis=1)]

    metrics = compute_model_metrics(df=df, y_pred=y_pred)

    # check some of the metrics
    assert np.round(metrics["weighted avg"]["precision"], 3) == 0.016
    assert np.round(metrics["weighted avg"]["recall"], 3) == 0.125
    assert np.round(metrics["weighted avg"]["f1-score"], 3) == 0.028
    assert metrics["weighted avg"]["support"] == 8


@pytest.mark.first
def test_check_performance():

    is_performant = check_performance(
        metrics={
            "GB": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "NL": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "ALL": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.1,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "FR": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "IT": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
        },
        cucumber_tests_df=pd.DataFrame(
            {
                TARGET_COLUMN: [{}],
            },
            index=[0],
        ),
        predictions_df=pd.DataFrame(
            {
                "labels": [{}],
            },
            index=[0],
        ),
    )
    assert ~is_performant

    is_performant = check_performance(
        metrics={
            "GB": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "NL": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "ALL": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "FR": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "IT": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
        },
        cucumber_tests_df=pd.DataFrame(
            {
                TARGET_COLUMN: [{}],
            },
            index=[0],
        ),
        predictions_df=pd.DataFrame(
            {
                "labels": [{}],
            },
            index=[0],
        ),
    )
    assert ~is_performant

    is_performant = check_performance(
        metrics={
            "GB": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "NL": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "ALL": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "FR": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
            "IT": {
                "Salary": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Refund": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Energy": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Internet_mobile": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Video_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Music_streaming": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Gym": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
                "Bonus_money": {
                    "support": 1000,
                    "recall": 0.8,
                    "precision": 0.8,
                    "f1-score": 0.8,
                },
            },
        },
        cucumber_tests_df=pd.DataFrame(
            {
                TARGET_COLUMN: [{}],
            },
            index=[0],
        ),
        predictions_df=pd.DataFrame(
            {
                "labels": [{}],
            },
            index=[0],
        ),
    )
    assert is_performant
