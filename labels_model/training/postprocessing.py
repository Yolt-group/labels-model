import tensorflow as tf

from labels_model.training.settings import (
    BUSINESS_RULES_DEFINITION,
    LABEL_DEPENDENCIES,
    TRAINING_LABELS,
    TRANSACTION_LABELS,
    N_TRAINING_LABELS,
    N_TRANSACTION_LABELS,
    GENERAL_SIMILARITY_THRESHOLD,
)


class BusinessRulesTransformer(tf.keras.layers.Layer):
    """
    Layer which applies business rules and modifies model similarities accordingly
    Note that the business logic is defined by the parameter BUSINESS_RULES_DEFINITION
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trainable = False

    def call(self, inputs, **kwargs):

        # extract similarities
        # >>> it will be modified on the fly while applying business rules
        postprocessed_similarities = inputs["similarities"]

        # generate [True] values mask with the shape of [n_rows, n_labels]
        # >>> by generating all_trues we make sure that we repeat the condition mask for every row
        all_trues = tf.cast(
            tf.clip_by_value(postprocessed_similarities, 1.0, 1.0), tf.dtypes.bool
        )

        # extract labels that have a business rule defined
        labels = BUSINESS_RULES_DEFINITION.keys()
        # iterate through all labels
        for label in labels:
            # extract label index mask
            label_index_mask = all_trues & tf.equal(TRAINING_LABELS, label)

            # iterate through the rules defined for given label
            for rule in BUSINESS_RULES_DEFINITION[label]:
                input_feature = inputs[rule["feature_name"]]

                # create a mask for the input based on the feature value & tensorflow operator defined in the rule
                tf_comparison_operator = rule["tf_comparison_operator"]

                input_feature_value_mask = all_trues & tf_comparison_operator(
                    tf.reshape(input_feature, (-1, 1)), rule["feature_value"]
                )

                # generate similarities with the value defined for the rule
                rule_similarity_value = tf.clip_by_value(
                    postprocessed_similarities,
                    rule["similarity_value"],
                    rule["similarity_value"],
                )

                # overwrite similarities with the value defined in the rule for given label
                postprocessed_similarities = tf.where(
                    tf.logical_and(label_index_mask, input_feature_value_mask),
                    rule_similarity_value,
                    postprocessed_similarities,
                )

        return postprocessed_similarities

    def compute_output_shape(self, input_shape):
        return input_shape[0], N_TRAINING_LABELS


class EliminateLabelDependencies(tf.keras.layers.Layer):
    """
    Eliminate the dependencies between labels. Within the conflicting labels, we leave the max predicted
    similarity unchanged and set -1 for the remaining conflicting labels.
    """

    def call(self, similarities):
        # generate [True] values mask with the shape of [n_rows, n_labels]
        # >>> by generating all_trues we make sure that we repeat the condition mask for every row
        all_trues = tf.cast(tf.clip_by_value(similarities, 1.0, 1.0), tf.bool)

        # iterate through conflicting labels
        for label_conflict in LABEL_DEPENDENCIES:
            # extract indices for conflicting labels
            conflicting_labels_indices = [
                TRAINING_LABELS.index(label) for label in label_conflict
            ]

            # extract similarities for conflicting labels based on their indices
            conflicting_labels_similarities = tf.gather(
                similarities, conflicting_labels_indices, batch_dims=-1
            )

            # extract best predicted similarity for conflicting labels
            conflicting_labels_best_predicted_similarity = tf.reshape(
                tf.reduce_max(conflicting_labels_similarities, axis=1), (-1, 1)
            )

            # extract mask for best predicted similarity within conflicting labels
            conflicting_labels_best_predicted_similarity_mask = tf.equal(
                similarities, conflicting_labels_best_predicted_similarity
            )

            # define mask for non conflicting labels
            not_conflicting_labels_index_flag = [
                True if label not in label_conflict else False
                for label in TRAINING_LABELS
            ]
            non_conflicting_labels_mask = all_trues & not_conflicting_labels_index_flag

            # leave similarities unchanged if it's the best prediction within the conflicting labels or it's about non conflicting labels;
            #   otherwise set the similarities to -1
            similarities = tf.where(
                tf.logical_or(
                    conflicting_labels_best_predicted_similarity_mask,
                    non_conflicting_labels_mask,
                ),
                similarities,
                tf.clip_by_value(similarities, -1.0, -1.0),
            )

        return similarities

    def compute_output_shape(self, input_shape):
        return input_shape[0], N_TRAINING_LABELS


class PredictionsIndicatorAboveThreshold(tf.keras.layers.Layer):
    """
    Return the matrix with 1. if the label similarity is above threshold and 0. otherwise
    The output shape would be (n_transactions, n_transaction_labels) including the label for "Unknown"
    """

    def __init__(self, **kwargs):
        super(PredictionsIndicatorAboveThreshold, self).__init__(**kwargs)

        # define list with indices for transaction labels to extract the similarities we are interested in
        self.transaction_labels_indices = [
            TRAINING_LABELS.index(label)
            for label in TRAINING_LABELS
            if label in TRANSACTION_LABELS
        ]

    def call(self, similarities):
        # extract similarities for transaction labels
        similarities_for_transaction_labels = tf.gather(
            similarities, self.transaction_labels_indices, batch_dims=-1
        )

        # create boolean where the similarity is equal or greater than the threshold
        postprocessed_similarities_above_threshold_mask = tf.greater_equal(
            similarities_for_transaction_labels, GENERAL_SIMILARITY_THRESHOLD
        )

        # convert boolean to float
        # > compiled Keras model objects always are attached to a loss function and the loss function must have floats (or similar) as inputs.
        postprocessed_similarities_above_threshold_float = tf.cast(
            postprocessed_similarities_above_threshold_mask, tf.dtypes.float32
        )

        return postprocessed_similarities_above_threshold_float

    def compute_output_shape(self, input_shape):
        return input_shape[0], N_TRANSACTION_LABELS
