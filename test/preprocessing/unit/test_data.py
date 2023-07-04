from labels_model.preprocessing.data import extract_users_sample
from pyspark.sql import functions as f


def test_create_training_data_frame(df):
    # due to sampling we test df shape within given boundaries
    assert len(df) > 0


def test_extract_users_sample(user_tables):
    users, test_users = user_tables

    user_sample = extract_users_sample(users=users, test_users=test_users)

    nr_NL_test_users = (
        user_sample.where(f.col("country_code") == "NL")
        .where(f.col("test_user") == 1)
        .count()
    )

    nr_total_test_users = user_sample.where(f.col("test_user") == 1).count()

    # there should only be NL test users
    # given our test data this should be at least 1 user
    assert nr_NL_test_users == nr_total_test_users > 0
