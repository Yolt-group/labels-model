"""
    Model Specific Settings

    For Data & Environment Specific settings use .yaml with yds
    For PyTest Specific settings use fixtures in pytest
"""

CATEGORY_LABELS = [
    "Housing",
    "Personal care",
    "Groceries",
    "Eating out",
    "Shopping",
    "Travel",
    "Transport",
    "Bills",
    "Transfers",
    "Cash",
    "Leisure",
    "Internal",
    "Income",
    "Charity",
    "Coffee",
    "Drinks",
    "Education",
    "Expenses",
    "Investments",
    "Lunch",
    "Gifts",
    "Kids",
    "Takeaway",
    "Petrol",
    "Rent",
    "Mortgage",
    "Utilities",
    "Vehicle",
    "Pets",
    "Savings",
]

TRANSACTION_LABELS = [
    "Salary",
    "Refund",
    "Energy",
    "Internet_mobile",
    "Video_streaming",
    "Music_streaming",
    "Gym",
    "Bonus_money",
]
TRAINING_LABELS = CATEGORY_LABELS + TRANSACTION_LABELS

# NOTE: due to mapping categories to integers, excluded labels must be assigned to the highest number
EXCLUDED_TRAINING_LABELS = ["General"]
LABELS = CATEGORY_LABELS + TRANSACTION_LABELS + EXCLUDED_TRAINING_LABELS

N_TRAINING_LABELS = len(TRAINING_LABELS)
N_LABELS = len(LABELS)
N_TRANSACTION_LABELS = len(TRANSACTION_LABELS)

# note that internal/debit transaction column is defined as numeric since its converted to 0/1 flag
NUMERIC_COLUMNS = ["amount", "transaction_type", "internal_transaction"]

PREPROCESSED_NUMERIC_COLUMNS = [
    "scaled_amount",
    "is_debit_transaction",
    "is_internal_transaction",
]

N_NUMERIC_COLUMNS = len(NUMERIC_COLUMNS)
INPUT_COLUMNS = ["description"] + NUMERIC_COLUMNS
TARGET_COLUMN = "target_label"
TARGET_COLUMN_INT = TARGET_COLUMN + "_int"
ADDITIONAL_TARGET_COLUMN_CATEGORIES = "target_category"
ADDITIONAL_TARGET_COLUMN_LABELS = "transaction_label"

UNIQUE_TRANSACTION_IDENTIFIER = [
    "user_id_hashed",
    "account_id_hashed",
    "transaction_id_hashed",
    "date",
]
