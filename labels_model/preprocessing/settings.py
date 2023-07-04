from labels_model.config.settings import *  # noqa: F403, F401

USER_FEEDBACK_EVENTS_COLUMNS_CATEGORIES = {
    "columns": [
        "id.userId",
        "id.accountId",
        "id.transactionId",
        "id.localDate",
        "id.pendingType",
        "time",
        "fact.category",
    ],
    "aliases": {
        "userId": "user_id",
        "accountId": "account_id",
        "transactionId": "transaction_id",
        "localDate": "date",
        "pendingType": "pending",
        "time": "feedback_time",
    },
}

USER_FEEDBACK_EVENTS_COLUMNS_LABELS = {
    "columns": [
        "entity__userId_hashed",
        "entity__accountId_hashed",
        "entity__transactionId_hashed",
        "creationTime",
        "entity__label",
        "year",
        "month",
    ],
    "aliases": {
        "entity__userId_hashed": "user_id_hashed",
        "entity__accountId_hashed": "account_id_hashed",
        "entity__transactionId_hashed": "transaction_id_hashed",
        "creationTime": "feedback_time",
        "entity__label": "transaction_label",
    },
}

TABLE_COLUMNS = {
    "users": {
        "columns": ["user_id_hashed", "country_code"],
    },
    "test_users": {"columns": ["user_id_hashed"]},
    "accounts": {
        "columns": [
            "user_id_hashed",
            "account_id_hashed",
            "deleted",
            "site_id",
        ],
    },
    "transactions": {
        "columns": [
            "user_id",
            "account_id",
            "transaction_id",
            "date",
            "pending",
            "description",
            "internal_transaction",
            "transaction_type",
            "amount",
            "cycle_id",
            "user_id_hashed",
            "account_id_hashed",
            "transaction_id_hashed",
            "month",
            "year",
        ]
    },
    "transaction_cycles": {
        "columns": ["user_id_hashed", "cycle_id", "model_base_period"]
    },
    "historical_categories_feedback": {
        "columns": [
            "user_id",
            "account_id",
            "transaction_id",
            "date",
            "pending",
            "feedback_time",
            "category",
        ]
    },
    "historical_labels_feedback": {
        "columns": [
            "user_id_hashed",
            "account_id_hashed",
            "transaction_id_hashed",
            "date",
            "pending",
            "feedback_time",
            "transaction_label",
        ]
    },
    "user_single_feedback_created_categories": USER_FEEDBACK_EVENTS_COLUMNS_CATEGORIES,
    "user_multiple_feedback_created_categories": USER_FEEDBACK_EVENTS_COLUMNS_CATEGORIES,
    "user_multiple_feedback_applied_categories": USER_FEEDBACK_EVENTS_COLUMNS_CATEGORIES,
    "user_feedback_labels": USER_FEEDBACK_EVENTS_COLUMNS_LABELS,
}

# ---------------------------
# Training data filters
# ---------------------------
COUNTRIES = ["GB", "FR", "IT", "NL"]
NL_SITE_IDS = [
    "b02fca30-65f6-470f-af1b-fbd5704abd56",
    "ed09586e-0f6d-44e3-8479-a5a35660f40b",
    "b17f3413-b84f-4495-8d5b-9ff1e840b7a6",
    "eedd41a8-51f8-4426-9348-314a18dbdec7",
    "c7987867-3219-4396-8d56-d79aff85a073",
    "1cc275ce-787f-45e1-bc0a-f8746cf5731b",
    "a6ccadc4-a2fa-11e9-a2a3-2a2ae2dbcce4",
    "6ed38773-34a6-4694-8e49-a50974f52510",
    "ee622d86-22cf-4f09-a475-198377971ff3",
    "2967f2c0-f0e6-4f1f-aeba-e4357b82ca7a",
    "bfdc30e3-1a08-4f2a-a85d-ac32c7227ccc",
    "2a609329-e2e8-44ac-9ee4-8896d68625ce",
    "7670247e-323e-4275-82f6-87f31119dbd3",
    "03433a5c-d1b1-41f7-b38f-119227bf7450",
    "44bbc1b2-029e-11e9-8eb2-f2801f1b9fd1",
    "13c00f7c-9746-11e9-bc42-526af7764f64",
]

# ---------------------------
# App configuration
# ---------------------------
PREPROCESSING_METADATA_FILE = "preprocessing_metadata.yaml"
