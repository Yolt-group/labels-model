from typing import AnyStr, Dict, Tuple, List

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    hamming_loss,
    label_ranking_average_precision_score,
)
from tensorflow.python.data.ops.dataset_ops import DatasetV1Adapter as TfDataset

from labels_model.training.postprocessing import (
    BusinessRulesTransformer,
    EliminateLabelDependencies,
    PredictionsIndicatorAboveThreshold,
)

from labels_model.training.settings import (
    PREPROCESSED_NUMERIC_COLUMNS,
    INPUT_COLUMNS,
    TARGET_COLUMN,
    TARGET_COLUMN_INT,
    N_TRAINING_LABELS,
    EXCLUDED_TRAINING_LABELS,
    WEIGHTED_METRICS_MIN_THRESHOLDS,
    SEQUENCE_LENGTH,
    VOCABULARY_SIZE,
    EMBEDDING_SIZE,
    BATCH_SIZE,
    MAX_EPOCHS,
    LEARNING_RATE,
    MARGIN,
    MAX_RECALL_RANK,
    TRAINING_LABELS,
    TRANSACTION_LABELS,
    N_NUMERIC_COLUMNS,
    LABELS,
    COUNTRIES,
)

from datascience_model_commons.utils import get_logger

logger = get_logger()


class TransactionLabelClassifier:
    def __init__(self, script_config: Dict):
        self.model = None
        self.postprocessed_model = None
        self.metrics = dict()
        self.metadata = dict()
        self.deploy_id = "1"

    def fit(self, df_train: pd.DataFrame, df_validation: pd.DataFrame):
        """Fit the model on a training data set and evaluate performance on a hold-out set"""

        # exclude given categories from train & validation dataset
        df_train_filtered = df_train[
            ~df_train[TARGET_COLUMN].isin(EXCLUDED_TRAINING_LABELS)
        ]
        df_validation_filtered = df_validation[
            ~df_validation[TARGET_COLUMN].isin(EXCLUDED_TRAINING_LABELS)
        ]

        # generate preprocessing parameters which creates global variables
        generate_preprocessing_parameters(df=df_train_filtered)

        # check missing labels on train and validation sets
        self.check_training_labels(df=df_train_filtered, df_name="Training set")
        self.check_training_labels(df=df_validation_filtered, df_name="Validation set")

        # convert pandas to preprocessed tf dataset
        train_dataset, validation_dataset = (
            raw_to_tf_preprocessed_dataset(
                df=df, batch_size=min(BATCH_SIZE, len(df)), shuffle_and_repeat=True
            )
            for df in [df_train_filtered, df_validation_filtered]
        )

        # build model
        self._build_model()

        # define callbacks for reducing learning rate and early stopping
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=2,
            min_lr=0.0001,
            verbose=0,
            min_delta=0.001,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, min_delta=0.001, verbose=0
        )

        # define steps per each epoch
        steps_per_epoch = np.clip(
            a=len(df_train_filtered) // BATCH_SIZE, a_min=1, a_max=None
        )
        validation_steps = np.clip(
            a=len(df_validation_filtered) // BATCH_SIZE, a_min=1, a_max=None
        )

        logger.debug(f"df_length: {df_train_filtered.shape}")

        # train model
        self.model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=MAX_EPOCHS,
            verbose=2,
            validation_data=validation_dataset,
            validation_steps=validation_steps,
            callbacks=[reduce_lr, early_stopping],
        )

        return self

    def save(self, *, path: AnyStr):
        input_amount = tf.keras.layers.Input(
            shape=(1,),
            dtype=tf.dtypes.float32,
            name="amount",
        )

        input_description = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.string, name="description"
        )

        # auxiliary inputs for business rules
        input_internal_transaction = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.string, name="internal_transaction"
        )
        input_transaction_type = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.string, name="transaction_type"
        )

        serving_inputs = {
            "description": input_description,
            "amount": input_amount,
            "internal_transaction": input_internal_transaction,
            "transaction_type": input_transaction_type,
        }

        preprocessing_layer = tf.keras.layers.Lambda(
            function=serving_fn,
            name="preprocessing_fn",
        )(serving_inputs)
        serving_output = self.postprocessed_model(preprocessing_layer)

        serving_model = tf.keras.Model(
            inputs=serving_inputs, outputs=serving_output, name="serving_model"
        )
        serving_model.compile(loss=dummy_loss)

        """Save tf model to tar gz file"""
        serving_model.save(filepath=f"{str(path)}/{self.deploy_id}")

    def predict_similarity(self, df: pd.DataFrame) -> np.array:
        """Return predicted similarities for each transaction label using the fitted model"""
        test_dataset = raw_to_tf_preprocessed_dataset(
            df=df, batch_size=len(df), shuffle_and_repeat=False
        )
        all_similarities = self.model.predict(test_dataset)

        # select only those similarities that are tied to transaction_labels rather than categories
        selected_indices = [
            i
            for i in range(len(TRAINING_LABELS))
            if TRAINING_LABELS[i] in TRANSACTION_LABELS
        ]

        return all_similarities[:, selected_indices]

    def predict(self, df: pd.DataFrame) -> np.array:
        """Return predicted final labels using the fitted model"""
        test_dataset = raw_to_tf_preprocessed_dataset(
            df=df,
            batch_size=len(df),
            shuffle_and_repeat=False,
        )
        return self.postprocessed_model.predict(test_dataset)

    def evaluate(self, df: pd.DataFrame) -> Dict:
        """Evaluate performance of fitted model on a validation data set"""

        # count rows and check if all classes are present in validation data
        self.metadata["n_test_samples"] = len(df)
        self.check_transaction_labels(df=df, df_name="Test set")

        # get predictions for test set
        y_pred = self.predict(df)

        self.metadata["metrics"] = metrics = compute_model_metrics(df=df, y_pred=y_pred)

        return metrics

    def _build_model(self):
        """Define model architecture"""

        # preprocessed input description - generate embeddings
        input_description = tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH,), dtype=tf.int32, name="preprocessed_description"
        )
        x = input_description
        # ADJUSTMENT: mask_zero=True added
        x = tf.keras.layers.Embedding(
            input_dim=VOCABULARY_SIZE,
            input_length=SEQUENCE_LENGTH,
            output_dim=EMBEDDING_SIZE,
            mask_zero=True,
        )(x)
        # ADJUSTMENT: nonzeromean replaced by zeromaskedaverage
        x = ZeroMaskedAverage(name="description_embedding")(x)
        description_embeddings = x

        # preprocessed numeric features
        input_numeric_features = tf.keras.layers.Input(
            shape=(N_NUMERIC_COLUMNS,),
            dtype=tf.float32,
            name="preprocessed_numeric_features",
        )

        # combine description and numeric features into transaction embedding
        transaction_features = tf.keras.layers.Concatenate()(
            [description_embeddings, input_numeric_features]
        )
        x = tf.keras.layers.Dense(units=EMBEDDING_SIZE, activation="linear")(
            transaction_features
        )
        x = tf.keras.layers.Lambda(
            lambda v: tf.nn.l2_normalize(v, axis=1), name="transaction_embeddings"
        )(x)
        transaction_embeddings = x

        # extract similarities between transactions & labels embeddings
        similarities = LabelEmbeddingSimilarity(name="similarities")(
            transaction_embeddings
        )

        # auxiliary inputs for business rules
        input_internal_transaction = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.float32, name="is_internal_transaction"
        )
        input_transaction_type = tf.keras.layers.Input(
            shape=(1,), dtype=tf.dtypes.float32, name="is_debit_transaction"
        )

        # model definition
        self.model = tf.keras.Model(
            inputs=[
                input_description,
                input_numeric_features,
                input_internal_transaction,
                input_transaction_type,
            ],
            outputs=similarities,
        )

        # define optimizer for loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # compile model with custom triplet loss
        self.model.compile(optimizer=optimizer, loss=triplet_loss)

        # use business rules transformer to overwrite similarities
        #   since we use model.output as input to postprocessing layer, it needs to be defined in the same function

        input_dict = {
            "similarities": self.model.output,
            "is_debit_transaction": input_transaction_type,
            "is_internal_transaction": input_internal_transaction,
        }

        postprocessed_similarities = BusinessRulesTransformer(
            name="postprocessed_similarities_business_rules"
        )(input_dict)

        # ensure that mutually exclusive labels are not assigned to the same transaction
        postprocessed_similarities = EliminateLabelDependencies(
            name="postprocessed_similarities_label_dependencies"
        )(postprocessed_similarities)

        # assign label if similarity is above threshold.
        predictions_indicator_above_threshold = PredictionsIndicatorAboveThreshold(
            name="predictions_indicator_above_threshold"
        )(postprocessed_similarities)

        # postprocessed model definition
        self.postprocessed_model = tf.keras.Model(
            inputs=self.model.input, outputs=predictions_indicator_above_threshold
        )

        # set "layer.trainable" to False in order to omit some "None gradient" errors while exporting the model
        # --- postprocessed model shouldn't be trainable
        for layer in self.postprocessed_model.layers:
            layer.trainable = False

        # compile postprocessed model
        #   note that model compilation is required to generate tf estimator; the loss function is used anywhere
        self.postprocessed_model.compile(loss=dummy_loss)

        return self

    @staticmethod
    def check_training_labels(*, df: pd.DataFrame, df_name: AnyStr = ""):
        """Check whether all labels are present in the training/validation data frame"""
        labels_present = df[TARGET_COLUMN].unique()
        label_difference = set(LABELS).difference(set(labels_present))
        n_missing = len(label_difference)

        if n_missing > 0:
            logger.warning(
                f"{df_name}: {n_missing} missing transaction labels: {label_difference}"
            )

    @staticmethod
    def check_transaction_labels(*, df: pd.DataFrame, df_name: AnyStr = ""):
        """Check whether all labels are present in the test data frame"""

        y_true = np.vstack(df[TARGET_COLUMN].apply(np.array))
        labels_present = list(np.max(y_true, axis=0))

        label_difference = [
            TRANSACTION_LABELS[i]
            for i in range(len(TRANSACTION_LABELS))
            if labels_present[i] == 0
        ]
        n_missing = len(label_difference)

        if n_missing > 0:
            logger.warning(
                f"{df_name}: {n_missing} missing transaction labels: {label_difference}"
            )


def preprocess_test_set(*, df: pd.DataFrame):

    # transform the targets to a form that can be used for multilabel evaluation
    test_targets = (
        df.groupby("unique_transaction_id").agg({TARGET_COLUMN: list}).reset_index()
    )
    mlb = MultiLabelBinarizer(classes=TRANSACTION_LABELS)
    test_targets[TARGET_COLUMN] = list(mlb.fit_transform(test_targets[TARGET_COLUMN]))

    # drop duplicates from test data and merge data back (if we would groupby INPUT_COLUMNS instead it drops all rows where there are nans)
    test_results = df[INPUT_COLUMNS + ["country_code", "unique_transaction_id"]]
    test_results = test_results.drop_duplicates()
    test_results[
        TARGET_COLUMN_INT
    ] = 99  # only required to run predict, but not actually used
    test_results = test_results.merge(
        test_targets, on="unique_transaction_id", how="inner"
    )

    return test_results


def generate_preprocessing_parameters(*, df: pd.DataFrame):
    """
    Generate parameters as global variables that are used in preprocessing datasets

    :param df: pandas dataframe with feature columns and target column
    :return: None
    """
    # generate global variable with parameters for scaling amount column
    #   NOTE that this variable has to be generated in the same script as preprocessing function since it's not
    #   shared across modules; global option is used to enable other functions in the same script use this variable
    global AMOUNT_SCALING_PARAMS
    AMOUNT_SCALING_PARAMS = {
        "min": np.float32(df["amount"].min()),
        "max": np.float32(df["amount"].max()),
    }

    # define parameters for missing values imputation
    global MISSING_VALUES_REPLACEMENT
    MISSING_VALUES_REPLACEMENT = {
        "description": "",
        "transaction_type": "",
        "amount": 0.0,
        "internal_transaction": "",
    }

    return None


def raw_to_tf_preprocessed_dataset(
    *, df: pd.DataFrame, batch_size: int, shuffle_and_repeat: bool = True
) -> TfDataset:
    """
    Convert numpy array to tf preprocessed dataset generator
    Recommended order of transformations (source: https://cs230.stanford.edu/blog/datapipeline):
        - create the dataset
        - shuffle
        - repeat
        - preprocess
        - batch
        - prefetch

    :param df: pandas dataframe with feature columns and target column
    :param batch_size: batch size
    :param shuffle_and_repeat: whether we should shuffle and repeat the dataset
    :return: tuple of (train, validation, test) tensorflow datasets
    """

    # replace missing values since Tensor does not accept None values and convert amount to float since spark generates Decimal()
    df_ = df.fillna(value=MISSING_VALUES_REPLACEMENT).astype({"amount": np.float32})

    # wrap dataframe to dictionary
    df_dict = dict(df_[INPUT_COLUMNS + [TARGET_COLUMN_INT]])

    # convert df to tensorflow dataset
    dataset = tf.data.Dataset.from_tensor_slices(df_dict)

    if shuffle_and_repeat:
        # shuffle rows with a buffer size equal to the length of the dataset; this ensures good shuffling
        #    for more details check: https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625
        shuffle_buffer_size = len(df)
        dataset = dataset.shuffle(shuffle_buffer_size)

        # repeat dataset elements
        #    since we do shuffle and then repeat we make sure that we always see every element in the dataset at each epoch
        dataset = dataset.repeat()

    # preprocess dataset
    #    use num_parallel_calls to parallelize
    dataset = dataset.map(
        lambda ds: (preprocessing_fn(inputs=ds), ds[TARGET_COLUMN_INT]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    # split dataset into batches
    #    use drop_remainder to not end up with the last batch having small number of rows
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # pre-fetching the data
    #    it will always have <buffer_size> batch ready to be loaded
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


@tf.function
def serving_fn(inputs: TfDataset):
    features = preprocessing_fn(inputs=inputs)

    # reshape is required for estimator to be exported correctly with known shapes
    features["preprocessed_description"] = tf.reshape(
        features["preprocessed_description"], (-1, SEQUENCE_LENGTH)
    )
    features["preprocessed_numeric_features"] = tf.reshape(
        features["preprocessed_numeric_features"], (-1, N_NUMERIC_COLUMNS)
    )

    # postprocessing features
    features["is_internal_transaction"] = tf.reshape(
        features["is_internal_transaction"], (-1, 1)
    )
    features["is_debit_transaction"] = tf.reshape(
        features["is_debit_transaction"], (-1, 1)
    )
    return features


def preprocessing_fn(*, inputs: TfDataset) -> TfDataset:
    """
    Preprocess tensorflow dataset

    :param inputs: tensorflow dataset with input columns
    :return: preprocessed tensorflow dataset
    """

    # initialize output dictionary
    outputs = dict()

    # preprocess description
    reshaped_description = tf.squeeze(
        tf.reshape(inputs["description"], (-1, 1)), axis=1
    )
    # ADJUSTMENT: pattern changed from r"[^a-zA-Z]+" to r"[^a-zA-Z\d]+|\d{2,}", so that we keep single numbers
    cleaned_description = tf.strings.lower(
        tf.strings.regex_replace(
            reshaped_description, pattern=r"[^a-zA-Z\d]+|\d{2,}", rewrite=" "
        )
    )
    # divide sequence length by 2 when splitting the description since by creating 1-ngrams and 2-grams we will double
    #    the sequence length
    tokens = tf.strings.split(
        cleaned_description,
        maxsplit=(SEQUENCE_LENGTH / 2 - 1),
    )

    ngrams = tf.strings.ngrams(data=tokens, ngram_width=(1, 2), separator=" ")
    token_indices = tf.strings.to_hash_bucket_fast(
        input=ngrams, num_buckets=VOCABULARY_SIZE
    ).to_tensor(shape=[None, SEQUENCE_LENGTH], default_value=0)
    token_indices_padded = tf.squeeze(input=token_indices)
    token_indices_padded.set_shape(SEQUENCE_LENGTH)
    outputs["preprocessed_description"] = token_indices_padded

    # preprocess numeric features
    numeric_features = dict()

    # scale amount
    numeric_features["scaled_amount"] = (
        inputs["amount"] - AMOUNT_SCALING_PARAMS["min"]
    ) / (AMOUNT_SCALING_PARAMS["max"] - AMOUNT_SCALING_PARAMS["min"])

    # debit transaction flag
    numeric_features["is_debit_transaction"] = tf.cast(
        tf.equal(inputs["transaction_type"], "debit"), tf.dtypes.float32
    )

    # internal transaction flag
    numeric_features["is_internal_transaction"] = tf.cast(
        tf.not_equal(inputs["internal_transaction"], ""), tf.dtypes.float32
    )

    # combine all numeric features
    # NOTE: the order is important! we extract the features based on the index in postprocessing layer
    preprocessed_numeric_features = tf.transpose(
        tf.squeeze(
            tf.stack(
                [numeric_features[col] for col in PREPROCESSED_NUMERIC_COLUMNS], axis=0
            )
        )
    )
    preprocessed_numeric_features.set_shape(N_NUMERIC_COLUMNS)
    outputs["preprocessed_numeric_features"] = preprocessed_numeric_features

    outputs["is_internal_transaction"] = numeric_features["is_internal_transaction"]
    outputs["is_debit_transaction"] = numeric_features["is_debit_transaction"]

    return outputs


class ZeroMaskedAverage(tf.keras.layers.Layer):
    """
    This layer is called after an Embedding layer.
    It zeros out all of the masked-out embeddings and returns the average of word embeddings.
    """

    def __init__(self, **kwargs):
        self.support_mask = True
        super(ZeroMaskedAverage, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[1]
        self.repeat_dim = input_shape[2]

    def call(self, embedding, mask=None):
        # create a mask
        mask = tf.cast(mask, "float32")
        mask = tf.keras.backend.repeat(mask, self.repeat_dim)
        mask = tf.keras.backend.permute_dimensions(mask, (0, 2, 1))

        # number of that rows are not all zeros
        number_of_non_zeros_elements = tf.reduce_sum(
            tf.cast(mask, tf.float32), axis=1, keepdims=False
        )

        # the mean of a zero-length vector is undefined, but for a quick and dirty fix we extract the max
        number_of_non_zeros_elements = tf.maximum(number_of_non_zeros_elements, 1.0)

        # extract the mean from word embeddings to create a description embedding
        average_embedding = (
            tf.reduce_sum(embedding * mask, axis=1, keepdims=False)
            / number_of_non_zeros_elements
        )

        return average_embedding


class LabelEmbeddingSimilarity(tf.keras.layers.Layer):
    """
    Layer used to generate similarity between label embeddings and transaction embeddings
    """

    def build(self, input_shape):
        self.label_embeddings = self.add_weight(
            "label_embeddings",
            shape=[N_TRAINING_LABELS, EMBEDDING_SIZE],
            initializer="uniform",
            trainable=True,
        )

        super(LabelEmbeddingSimilarity, self).build(input_shape)

    def call(self, transaction_embeddings):
        label_embeddings = tf.nn.l2_normalize(self.label_embeddings, axis=1)
        similarities = tf.matmul(
            transaction_embeddings, label_embeddings, transpose_b=True
        )

        return similarities

    def compute_output_shape(self, input_shape):
        return input_shape[0], N_TRAINING_LABELS


def triplet_loss(labels, similarities):
    """
    Triplet loss function

    :param labels: target column labels
    :param similarities: dot product of text and label embeddings
    :return: loss value
    """
    # extract positive and all negative examples
    # - positive example is for target category; negative is for all other categories
    labels_indices = tf.cast(labels, tf.int32)
    pos = tf.gather(similarities, labels_indices, batch_dims=1)
    negatives_mask = tf.squeeze(
        1 - tf.one_hot(labels_indices, depth=N_TRAINING_LABELS, axis=1), axis=2
    )
    neg = tf.reshape(
        tf.boolean_mask(similarities, negatives_mask),
        (tf.shape(negatives_mask)[0], N_TRAINING_LABELS - 1),
    )

    # select the negative example that has the smallest similarity
    smallest_neg_similarity_tiled = tf.multiply(
        tf.ones_like(neg), tf.reduce_min(neg, axis=1, keepdims=True)
    )

    # if a similarity for negative example is lower than for positive example -> take neg;
    # else if negative example has higher similarity than positive example -> take the one with the smallest similarity
    neg_similarity_smaller_than_pos = tf.where(
        tf.less(neg, pos), neg, smallest_neg_similarity_tiled
    )

    # as final negative example, select the 'closest' negative example to the positive example
    semi_hard_neg = tf.reduce_max(
        neg_similarity_smaller_than_pos, axis=1, keepdims=True
    )

    # minimize the distance between the positive example and 'closest' negative example + margin
    loss = tf.reduce_mean(tf.maximum(0.0, -(pos - semi_hard_neg) + MARGIN))

    return loss


def dummy_loss(y_true, y_pred):
    """
    Dummy loss function which returns 0; its required for the compilation of postprocessed model

    :param y_true: target column labels
    :param y_pred: predictions
    :return: loss value
    """
    # note that when using tf.constant([0.]) there is an error that Variable has `None` for gradient
    #  therefore multiplication of y_pred with 0. is used
    loss = y_pred * 0.0

    return loss


def compute_model_metrics(df: pd.DataFrame, y_pred: np.array) -> Dict:
    """Compute model metrics: classification"""

    # assumes that the inputed df has transformed target labels e.g. [0,0], [0,1], [1,1] etc.

    y_true = np.vstack(df[TARGET_COLUMN].apply(np.array))

    # generate multilabel classification metrics
    metrics = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=TRANSACTION_LABELS,
        zero_division=0,
        output_dict=True,
    )

    # generate multilabel metrics
    metrics["subset_accuracy"] = np.round(accuracy_score(y_true, y_pred), 4)
    metrics["hamming_loss"] = np.round(hamming_loss(y_true, y_pred), 4)

    # generate coverage metric - fraction of tx that has any label
    n_with_label = metrics["weighted avg"]["support"]
    n_total = len(df)
    metrics["coverage"] = np.round(n_with_label / n_total, 4)

    return metrics


def ranking_metrics(y_true_int: np.array, y_score: np.array) -> Tuple[float, List]:
    """Compute ranking metrics: mean reciprocal rank & recall at each rank until MAX_RANK define in the settings"""

    # transform true value to the array of values with 1. in the index of given category
    #   i.e. y_true=3 is changed to y_true=[0.,0.,0.,1.,...] with the length of number of categories
    y_true_int_transformed = np.zeros_like(y_score)
    y_true_int_transformed[np.arange(len(y_score)), y_true_int] = 1.0

    # label_ranking_average_precision_score is equal to mean reciprocal rank since there is exactly one relevant
    #   label per given user and partner
    mean_reciprocal_rank = label_ranking_average_precision_score(
        y_true=y_true_int_transformed,
        y_score=y_score,
    )

    # sort arguments to extract top n recommendations; minus score is used for descending order
    # i.e. for given list of scores y_score = [0.6, 0.5, 0.7], the output would be y_rank_ind = [1, 2, 0]
    y_rank_ind = np.argsort(-y_score, axis=1)

    # sort y_true based on score at rank n
    n_relevant_at_rank = np.take_along_axis(
        y_true_int_transformed, y_rank_ind, axis=1
    ).cumsum(axis=1)

    # calculate recall for each rank with a limit of max recall rank
    recall_at_rank = n_relevant_at_rank[:, : (MAX_RECALL_RANK + 1)] / 1.0

    return mean_reciprocal_rank, recall_at_rank


def calculate_description_length_distribution(df: pd.DataFrame) -> Dict:
    """
    Compute histogram of number of characters in input description

    :param df: data frame containing input data
    :return: dictionary containing histogram {bucket_right_edge: probability}
    """

    n_samples = len(df)
    buckets = list(range(0, 260, 10))
    input_description_length_counts = np.histogram(
        df["description"].str.len(), bins=buckets
    )[0]

    input_description_length_distribution = dict(
        zip(buckets[1:], (input_description_length_counts / n_samples).tolist())
    )

    return input_description_length_distribution


def create_cucumber_test_sample() -> pd.DataFrame:
    """
    Generate test sample for cucumber tests; passing all test is required to make model performant

    :return: pandas dataframe with input columns for the model and target category
    """
    test_sample = (
        pd.DataFrame(
            columns=[
                "unique_transaction_id",
                "description",
                "amount",
                "transaction_type",
                "internal_transaction",
                "country_code",
                TARGET_COLUMN,
                "target_label_int",
                "bank_counterparty_iban",
            ]
        )
        # Salary payment example
        .append(
            [
                {
                    "unique_transaction_id": "1111",
                    "description": "Salary payment wuhuuu",
                    "amount": 250.0,
                    "transaction_type": "credit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Salary"},
                }
            ],
            ignore_index=True,
        )
        # Debit salary payment, shouldn't be salary
        .append(
            [
                {
                    "unique_transaction_id": "1112",
                    "description": "Salary",
                    "amount": 250.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
        # Salary with low amount [not working at the moment]
        .append(
            [
                {
                    "unique_transaction_id": "1113",
                    "description": "amount too low but still somehow salary",
                    "amount": 10.0,
                    "transaction_type": "credit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Salary"},
                }
            ],
            ignore_index=True,
        )
        # Refund example
        .append(
            [
                {
                    "unique_transaction_id": "1114",
                    "description": " amazon Refund",
                    "amount": 10.0,
                    "transaction_type": "credit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Refund"},
                }
            ],
            ignore_index=True,
        )
        # Debit refund, shouldn't be refund
        .append(
            [
                {
                    "unique_transaction_id": "1115",
                    "description": "Refund",
                    "amount": 10.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
        # Both salary and refund amount and keywords are closer to salary so should be salary
        .append(
            [
                {
                    "unique_transaction_id": "1116",
                    "description": "Salary refun but not really",
                    "amount": 1500.0,
                    "transaction_type": "credit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Salary"},
                }
            ],
            ignore_index=True,
        )
        # Energy example
        .append(
            [
                {
                    "unique_transaction_id": "1117",
                    "description": "Naam: Vattenfall Klantenservice",
                    "amount": 100.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Energy"},
                }
            ],
            ignore_index=True,
        )
        # Internet_mobile example
        .append(
            [
                {
                    "unique_transaction_id": "1118",
                    "description": "EE & T-MOBILE",
                    "amount": 21.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Internet_mobile"},
                }
            ],
            ignore_index=True,
        )
        # Video streaming example
        .append(
            [
                {
                    "unique_transaction_id": "1119",
                    "description": "NETFLIX.COM",
                    "amount": 10.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Video_streaming"},
                }
            ],
            ignore_index=True,
        )
        # Music streaming example
        .append(
            [
                {
                    "unique_transaction_id": "1120",
                    "description": "Amazon Music",
                    "amount": 10.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Music_streaming"},
                }
            ],
            ignore_index=True,
        )
        # Music Streaming example
        .append(
            [
                {
                    "unique_transaction_id": "1121",
                    "description": "SPOTIFY UK",
                    "amount": 10.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Music_streaming"},
                }
            ],
            ignore_index=True,
        )
        # Gym example
        .append(
            [
                {
                    "unique_transaction_id": "1122",
                    "description": "Classpass Monthly, Classpass.com",
                    "amount": 15.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Gym"},
                }
            ],
            ignore_index=True,
        )
        # Bonus money example
        .append(
            [
                {
                    "unique_transaction_id": "1123",
                    "description": "amazon",
                    "amount": 100.0,
                    "transaction_type": "credit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: {"Bonus_money"},
                }
            ],
            ignore_index=True,
        )
        # Debit bonus money, shouldn't be bonus money
        .append(
            [
                {
                    "unique_transaction_id": "1125",
                    "description": "WWW.SKYBET.COM",
                    "amount": 100.0,
                    "transaction_type": "debit",
                    "internal_transaction": "",
                    "country_code": "GB",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
        # Internal bonus money
        .append(
            [
                {
                    "unique_transaction_id": "1126",
                    "description": "WWW.SKYBET.COM",
                    "amount": 100.0,
                    "transaction_type": "credit",
                    "internal_transaction": "ABC",
                    "country_code": "GB",
                    TARGET_COLUMN: set(),
                }
            ],
            ignore_index=True,
        )
    ).fillna("")

    return test_sample


def check_performance(
    *, metrics: Dict, cucumber_tests_df: pd.DataFrame, predictions_df: pd.DataFrame
) -> bool:
    """
    Verify if critical performance metrics are above some preset threshold and whether cucumber tests pass.

    :param predictions_df: predictions as model output
    :param metrics: dictionary containing model performance metrics which is the output of model.evaluate
        therefore we use predefined keys below
    :param cucumber_tests_df: cucumber test dataframe with predictions and expected category
    :return: boolean indicating if standards are met
    """
    # unpack model predictions

    cucumber_tests_passed = True

    for i in range(0, len(cucumber_tests_df)):
        logger.info(
            f"Cucumber test no {i+1}: Target = {cucumber_tests_df.loc[i, TARGET_COLUMN]}, Prediction = {predictions_df.loc[i, 'labels']}"
        )

        if cucumber_tests_df.loc[i, TARGET_COLUMN] != predictions_df.loc[i, "labels"]:
            logger.warning("Cucumber tests failing")
            cucumber_tests_passed = False
            break

    # extract weighted average metrics and compare with thresholds
    for country in COUNTRIES.keys():
        for label in TRANSACTION_LABELS:
            weighted_average_metrics = metrics[country][label]
            metrics_above_thresholds = all(
                weighted_average_metrics[key] >= (WEIGHTED_METRICS_MIN_THRESHOLDS[key])
                for key in WEIGHTED_METRICS_MIN_THRESHOLDS
            )

    if not metrics_above_thresholds:
        logger.warning("Weighted recall, precision or f1-score below threshold")

    return cucumber_tests_passed & metrics_above_thresholds
