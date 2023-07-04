from labels_model.config.settings import *  # noqa: F401, F403
import tensorflow as tf

# Data samples size configuration
N_VALIDATION_SAMPLES = 100000
N_TEST_SAMPLES = 100000

# Model parameters
BATCH_SIZE = 512
SEQUENCE_LENGTH = 60
VOCABULARY_SIZE = 600000
EMBEDDING_SIZE = 32

MAX_EPOCHS = 40
LEARNING_RATE = 0.01
MARGIN = 0.1

GENERAL_SIMILARITY_THRESHOLD = 0.55

# Performance metrics configuration
MAX_RECALL_RANK = 3
WEIGHTED_METRICS_MIN_THRESHOLDS = {"precision": 0.60, "recall": 0.60, "f1-score": 0.60}
COVERAGE_MIN_THRESHOLD = 0.80

# Model output configuration
MODEL_ARTIFACT_FILE = "transaction-label-model"
MODEL_METADATA_FILE = "training_metadata.yaml"

BUSINESS_RULES_DEFINITION = {
    "Salary": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_debit_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        }
    ],
    "Refund": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_debit_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
    "Energy": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
    "Internet_mobile": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
    "Video_streaming": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
    "Music_streaming": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
    "Gym": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
    "Bonus_money": [
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_debit_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
        {
            "feature_set": PREPROCESSED_NUMERIC_COLUMNS,
            "feature_name": "is_internal_transaction",
            "feature_value": 1.0,
            "tf_comparison_operator": tf.equal,
            "similarity_value": -1.0,
        },
    ],
}

# which labels should never be assigned to the same transaction
LABEL_DEPENDENCIES = [
    ("Salary", "Refund"),
    ("Salary", "Energy"),
    ("Salary", "Internet_mobile"),
    ("Salary", "Video_streaming"),
    ("Salary", "Music_streaming"),
    ("Salary", "Gym"),
    ("Salary", "Bonus_money"),
]

COUNTRIES = {
    "GB": ["GB"],
    "FR": ["FR"],
    "IT": ["IT"],
    "NL": ["NL"],
    "ALL": ["GB", "FR", "IT", "NL"],
}
