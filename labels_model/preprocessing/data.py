import datetime
from pathlib import Path
from typing import Tuple, AnyStr
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql import Window
from sklearn.model_selection import StratifiedShuffleSplit

from datascience_model_commons.utils import get_logger
from datascience_model_commons.deploy.config.domain import YDSProjectConfig

from datascience_model_commons.spark import read_data

from labels_model.preprocessing.settings import (
    TABLE_COLUMNS,
    EXCLUDED_TRAINING_LABELS,
    UNIQUE_TRANSACTION_IDENTIFIER,
    INPUT_COLUMNS,
    TARGET_COLUMN,
    ADDITIONAL_TARGET_COLUMN_CATEGORIES,
    ADDITIONAL_TARGET_COLUMN_LABELS,
    NL_SITE_IDS,
    LABELS,
    TARGET_COLUMN_INT,
)

logger = get_logger()


def list_folders_on_s3(*, path: AnyStr) -> list:
    """
    Function that creates a list with folders under given path on s3

    :param path: s3a path where to list folders
    :return: a list of folders in given path on s3
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects")

    # split path into bucket and prefix
    # - since s3 paths are URIs, we can use urlparse to extract bucket and prefix
    #    urlparse transform path into: <scheme>://<netloc><path>
    url_path = urlparse(path)
    bucket, prefix = (url_path.netloc, url_path.path)

    # extract prefix parent since the path is directly to '*' or '*.csv'
    # - note that the prefix includes slash at the beginning therefore [1:] is used
    # - note that the prefix directory has to end up with slash
    prefix_directory = f"{Path(prefix[1:]).parent}/"

    # append folders to the list
    folders_list = []
    for result in paginator.paginate(
        Bucket=bucket, Prefix=prefix_directory, Delimiter="/"
    ):
        for prefix in result.get("CommonPrefixes", []):
            full_prefix_path = Path(prefix.get("Prefix"))
            folder_name = full_prefix_path.name
            folders_list.append(folder_name)

    return folders_list


def read_data_and_select_columns(
    table: AnyStr,
    spark: SparkSession,
    project_config: YDSProjectConfig,
) -> pyspark.sql.DataFrame:
    """
    Function that reads data and selects the relevant columns

    :param table: name of the table that should be read
    :param spark: spark session
    :param config: categories configuration
    :return: pyspark table with relevant columns
    """

    # extract table path from categories configuration

    script_config = project_config.preprocessing.script_config

    data_file_paths = script_config.get("data_file_paths")

    # print("old: ")
    # file_path = config.__getattribute__(f"{table}_path")

    file_path = data_file_paths[table]

    # extract columns & aliases it exists from selected table
    columns = TABLE_COLUMNS[table]["columns"]
    aliases = TABLE_COLUMNS[table].get("aliases", False)

    # read data and repartition
    df = read_data(file_path=file_path, spark=spark).select(columns)

    # if columns need to be renamed, do so
    if aliases:
        for column_name, alias in aliases.items():
            df = df.withColumnRenamed(column_name, alias)

    logger.info(f"{table}: {file_path}")

    return df


def create_training_data_frame(
    *,
    transactions: pyspark.sql.DataFrame,
    accounts: pyspark.sql.DataFrame,
    users: pyspark.sql.DataFrame,
    test_users: pyspark.sql.DataFrame,
    user_single_feedback_created_categories: pyspark.sql.DataFrame,
    user_multiple_feedback_created_categories: pyspark.sql.DataFrame,
    user_multiple_feedback_applied_categories: pyspark.sql.DataFrame,
    historical_categories_feedback: pyspark.sql.DataFrame,
    historical_labels_feedback: pyspark.sql.DataFrame,
    start_training_date=datetime,
    nl_start_training_date=datetime,
    end_training_date=datetime,
    n_labels_feedback: int,
    n_categories_feedback: int,
    n_validation_samples: int,
    n_test_samples: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create a data frame for model training

    :param transactions: PySpark DataFrame referring to the raw transactions table
    :param accounts: PySpark DataFrame referring to the raw account table
    :param users: PySpark DataFrame referring to the raw users table
    :param test_users: PySpark DataFrame referring to the interim table including all test users
    :param user_single_feedback_created_categories: PySpark DataFrame referring to the data science events with single feedback
    :param user_multiple_feedback_created_categories: PySpark DataFrame referring to the data science events with multiple feedback
            created - only includes the transaction on which user clicked
    :param user_multiple_feedback_applied_categories: PySpark DataFrame referring to the data science events with multiple feedback
            applied - includes the transaction(s) for which multiple feedback rule was applied
    :param user_feedback_labels: PySpark DataFrame referring to the data science events with single feedback
    :param historical_categories_feedback: PySpark DataFrame referring to the all tables with static feedback
    :param historical_labels_feedback: PySpark DataFrame referring to the all tables with static feedback for labels
    :param start_training_date: starting training date.
    :param n_labels_feedback: number of max feedback wanted from per label per country
    :param n_categories_feedback: number of total categories feedback wanted in the model
    :param n_validation_samples: number of validation samples
    :param n_test_samples: number of test samples
    :return: tuple containing filtered Pandas DataFrame and sample mimicking production data
    """

    # extract users sample
    df_users_sample = extract_users_sample(users=users, test_users=test_users)

    # extract non deleted accounts
    df_non_deleted_accounts = extract_non_deleted_accounts(accounts=accounts)

    transaction_base = extract_transactions(
        user=df_users_sample,
        account=df_non_deleted_accounts,
        transaction=transactions,
        start_date=start_training_date,
        nl_start_date=nl_start_training_date,
        end_date=end_training_date,
    ).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    logger.info(f"{transaction_base.count()} Transactions extracted")

    # extract the transaction_labels feedback data
    df_labels_feedback = extract_labels_feedback(
        historical_feedback=historical_labels_feedback,
        user=df_users_sample,
        transaction=transaction_base,
        n_samples=n_labels_feedback,
    ).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    logger.info(f"{df_labels_feedback.count()} Labels feedback extracted")

    # extract the category feedback data
    df_categories_feedback = extract_categories_feedback(
        user_single_feedback_created=user_single_feedback_created_categories,
        user_multiple_feedback_created=user_multiple_feedback_created_categories,
        user_multiple_feedback_applied=user_multiple_feedback_applied_categories,
        historical_feedback=historical_categories_feedback,
        transaction=transaction_base,
        n_samples=n_categories_feedback,
    ).persist(pyspark.StorageLevel.MEMORY_AND_DISK)

    logger.info(f"{df_categories_feedback.count()} categories feedback extracted")

    transaction_base.unpersist()

    # extract final users & accounts sample
    df_training_sample = (
        df_categories_feedback.join(
            df_labels_feedback,
            on=UNIQUE_TRANSACTION_IDENTIFIER + INPUT_COLUMNS + ["country_code"],
            how="outer",
        )
        .fillna(EXCLUDED_TRAINING_LABELS[0], subset=ADDITIONAL_TARGET_COLUMN_CATEGORIES)
        .fillna("", subset=["description"])
        .select(
            INPUT_COLUMNS
            + ["country_code"]
            + [ADDITIONAL_TARGET_COLUMN_CATEGORIES]
            + [TARGET_COLUMN]
        )
        .withColumn("unique_transaction_id", f.monotonically_increasing_id())
    ).toPandas()

    train, validation, test = split_training_data_frame(
        df=df_training_sample,
        n_validation_samples=n_validation_samples,
        n_test_samples=n_test_samples,
    )

    return train, validation, test


def extract_users_sample(
    *, users: pyspark.sql.DataFrame, test_users: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Extract users sample based on predefined countries and excluding test users

    :param users: PySpark DataFrame referring to the raw users table
    :param test_users: PySpark DataFrame referring to the interim table including all test users
    :return: final users sample
    """
    non_nl_test_users = (f.col("country_code") != "NL") & (f.col("test_user") == 1)

    # define rule for including country specific users
    df_users_sample = (
        users.join(
            test_users.select("user_id_hashed", f.lit(1).alias("test_user")),
            "user_id_hashed",
            "left",
        )
        .fillna(0, subset=["test_user"])
        .where(~non_nl_test_users)
    )

    return df_users_sample


def extract_non_deleted_accounts(
    *, accounts: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Extract accounts sample excluding deleted accounts

    :param accounts: PySpark DataFrame referring to the raw account table
    :return: final accounts sample
    """

    # define rule for including non deleted accounts
    non_deleted_accounts = f.col("deleted").isNull()

    df_non_deleted_accounts = accounts.where(non_deleted_accounts)

    return df_non_deleted_accounts


def extract_transactions(
    *,
    user: pyspark.sql.DataFrame,
    account: pyspark.sql.DataFrame,
    transaction: pyspark.sql.DataFrame,
    start_date: datetime,
    nl_start_date: datetime,
    end_date: datetime,
) -> pyspark.sql.DataFrame:
    """
    Extract trx sample based on predefined countries and excluding test users

    :param transaction: PySpark DataFrame referring to the raw transaction table
    :return: final transactions sample
    """
    transactions_date_filter = (
        ~(f.col("country_code") == "NL")
        & ((f.col("year") >= start_date.year) & (f.col("month") >= start_date.month))
        & ((f.col("year") <= end_date.year))
    ) | (
        (f.col("country_code") == "NL")
        & (
            (f.col("year") >= nl_start_date.year)
            & (f.col("month") >= nl_start_date.month)
        )
        & ((f.col("year") <= end_date.year))
    )

    non_NL_accounts_for_NL_users = (f.col("country_code") == "NL") & ~(
        f.col("site_id").isin(NL_SITE_IDS)
    )

    # define rule for including country specific users
    transaction_base = (
        user.join(
            account,
            on="user_id_hashed",
            how="inner",
        )
        .join(
            transaction,
            on=["user_id_hashed", "account_id_hashed"],
            how="inner",
        )
        .where(transactions_date_filter)
        .where(~non_NL_accounts_for_NL_users)
    )

    return transaction_base


def extract_categories_feedback(
    *,
    user_single_feedback_created: pyspark.sql.DataFrame,
    user_multiple_feedback_created: pyspark.sql.DataFrame,
    user_multiple_feedback_applied: pyspark.sql.DataFrame,
    historical_feedback: pyspark.sql.DataFrame,
    transaction: pyspark.sql.DataFrame,
    n_samples: int,
) -> pyspark.sql.DataFrame:
    """
    Extract target category for recent feedback date

    :param user_single_feedback_created: PySpark DataFrame referring to the data science events with single feedback
    :param user_multiple_feedback_created: PySpark DataFrame referring to the data science events with multiple feedback
            created - only includes the transaction on which user clicked
    :param user_multiple_feedback_applied: PySpark DataFrame referring to the data science events with multiple feedback
            applied - includes the transaction(s) for which multiple feedback rule was applied
    :param historical_feedback: PySpark DataFrame referring to the static feedback tables
    :param user: user table in order to obtain country information
    :param account: account table in order to exclude feedback from already deleted accounts
    :param n_samples: maximum number of labels feedback wanted from per label per country
    :return: final table with the most recent feedback category assigned to the transaction
    """

    # define feedback columns so that we make sure that the order of the columns in each table is the same
    feedback_columns = [
        "user_id",
        "account_id",
        "transaction_id",
        "date",
        "feedback_time",
        "category",
    ]

    # append all feedback tables
    feedback_combined = (
        historical_feedback.select(feedback_columns)
        .union(user_single_feedback_created.select(feedback_columns))
        .union(user_multiple_feedback_created.select(feedback_columns))
        .union(user_multiple_feedback_applied.select(feedback_columns))
        # since there are some duplicates in events data, we make sure to remove them
        .dropDuplicates()
    )

    # select last feedback time for given transaction
    window_over_transaction = Window.partitionBy(
        "user_id", "account_id", "transaction_id", "date"
    ).orderBy(f.desc("feedback_time"))
    most_recent_feedback = f.col("row_nr") == 1

    # some synthetic data is missing feedback_time but that is around 2020-12-01 so fill it
    df_last_feedback = feedback_combined.withColumn(
        "row_nr", f.row_number().over(window_over_transaction)
    ).where(most_recent_feedback)

    # filter transactions for training after defined training date

    df_filtered = (
        df_last_feedback.join(
            transaction,
            on=["user_id", "account_id", "transaction_id", "date"],
            how="inner",
        )
        .withColumnRenamed("category", ADDITIONAL_TARGET_COLUMN_CATEGORIES)
        .select(
            UNIQUE_TRANSACTION_IDENTIFIER
            + INPUT_COLUMNS
            + ["country_code"]
            + [ADDITIONAL_TARGET_COLUMN_CATEGORIES]
        )
    )

    df_categories_feedback = sample_pyspark_df_to_pandas_df(
        df=df_filtered,
        n_samples=n_samples,
    )

    return df_categories_feedback


def extract_labels_feedback(
    *,
    historical_feedback: pyspark.sql.DataFrame,
    user: pyspark.sql.DataFrame,
    transaction: pyspark.sql.DataFrame,
    n_samples: int,
) -> pyspark.sql.DataFrame:
    """
    Extract target category for recent feedback date

    :param user_single_feedback_created: PySpark DataFrame referring to the data science events with single feedback
    :param historical_feedback: PySpark DataFrame referring to the static feedback tables
    :param user: user table in order to obtain country information
    :param account: account table in order to exclude feedback from already deleted accounts
    :param start_training_date: date for filtering desired feedback date range
    :param n_samples: maximum number of labels feedback wanted from per label per country
    :return: final table with the most recent feedback category assigned to the transaction
    """

    # define feedback columns so that we make sure that the order of the columns in each table is the same
    feedback_columns = [
        "user_id_hashed",
        "account_id_hashed",
        "transaction_id_hashed",
        "date",
        "feedback_time",
        "transaction_label",
    ]

    # filter transactions for training after defined training date
    df_filtered = historical_feedback.join(
        user,
        on="user_id_hashed",
        how="inner",
    ).select(feedback_columns + ["country_code"])

    # Sample labels with a maximum number, include all if the available data is under this maximum limit
    df_sampled_labels = sample_labels(df=df_filtered, n_samples=n_samples)

    df_labels_processed = process_labels_feedback(
        last_labels_feedback=df_sampled_labels,
    )

    df_labels_feedback = transaction.join(
        df_labels_processed.drop("country_code"),
        on=UNIQUE_TRANSACTION_IDENTIFIER,
        how="inner",
    ).select(
        UNIQUE_TRANSACTION_IDENTIFIER
        + INPUT_COLUMNS
        + ["country_code"]
        + [TARGET_COLUMN]
    )

    return df_labels_feedback


def process_labels_feedback(
    *, last_labels_feedback: pyspark.sql.DataFrame
) -> pyspark.sql.DataFrame:
    """
    Some extra processing of the labels feedback.

    :param last_labels_feedback: PySpark DataFrame referring to the static feedback tables
    :return: final table containing one row per transaction and a list of all transaction labels in the target_column
    """

    condition1 = f.array_contains(TARGET_COLUMN, "Salary") & (f.col("n_labels") > 1)
    condition2 = f.array_contains(TARGET_COLUMN, "Energy") & f.array_contains(
        TARGET_COLUMN, "Internet_mobile"
    )

    df_last_labels_feedback = (
        last_labels_feedback.groupBy(UNIQUE_TRANSACTION_IDENTIFIER + ["country_code"])
        .agg(
            f.collect_list(ADDITIONAL_TARGET_COLUMN_LABELS).alias(TARGET_COLUMN),
            f.countDistinct(ADDITIONAL_TARGET_COLUMN_LABELS).alias("n_labels"),
        )
        .where(~condition1)
        .where(~condition2)
        .drop("n_labels")
    )

    return df_last_labels_feedback


def sample_pyspark_df_to_pandas_df(
    *, df: pyspark.sql.DataFrame, n_samples: int
) -> pd.DataFrame:
    """
    Extract the sample of pyspark dataframe and save it as pandas df

    :param df: PySpark DataFrame
    :param n_samples: number of samples to extract
    :return: Data sample in Pandas DataFrame
    """

    df_rows = df.count()

    df_sample_ratio = np.min(
        [n_samples / df_rows, 1.0]
    )  # make sure the ratio is not higher than 1 - it may happen that for some countries we have smaller sample
    df_sample = df.sample(withReplacement=False, fraction=df_sample_ratio, seed=123)

    return df_sample


def sample_labels(*, df: pyspark.sql.DataFrame, n_samples: int) -> pd.DataFrame:
    """
    Sample by group instead of sample per label per country, runs more efficiently
    :param df: PySpark DataFrame
    :param n_samples: number max samples desired per label per country
    :return: Data sample in Pandas DataFrame
    """

    sampling_ratio = f.col("max_sample_size") / f.col("n_rows")

    # Combine identifiers as function accepts one column for sample group
    df_with_sample_column = df.withColumn(
        "sampler", f.concat(f.col("country_code"), f.col("transaction_label"))
    )

    # calculate group counts and sampling ratios per group
    df_sample_size = (
        (
            df_with_sample_column.groupBy("sampler")
            .agg(f.count("transaction_id_hashed").alias("n_rows"))
            .withColumn("max_sample_size", f.lit(n_samples))
            .withColumn(
                "sampling_ratios",
                f.when(sampling_ratio >= 1, 1).otherwise(sampling_ratio),
            )
            .select("sampler", "sampling_ratios")
        )
        .toPandas()
        .set_index("sampler")
        .to_dict()
    )

    df_sample = df_with_sample_column.sampleBy(
        "sampler", fractions=df_sample_size["sampling_ratios"], seed=42
    )

    return df_sample


def split_training_data_frame(
    *, df: pd.DataFrame, n_validation_samples: int, n_test_samples: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split Pandas DataFrame in three stratified sets

    :param df: PySpark DataFrame to split
    :param n_validation_samples: number of rows in the validation frame
    :param n_test_samples: number of rows in the test frame
    :return: tuple of (train, validation, test) pandas datasets
    """

    # split dataset into train & test
    split_train_test = StratifiedShuffleSplit(
        n_splits=1, test_size=n_test_samples, random_state=42
    )

    train_index, test_index = next(
        split_train_test.split(df, df[ADDITIONAL_TARGET_COLUMN_CATEGORIES])
    )
    df_train_val = df.iloc[train_index]
    df_test = df.iloc[test_index]

    # split train dataset into train & validation
    split_train_val = StratifiedShuffleSplit(
        n_splits=1, test_size=n_validation_samples, random_state=42
    )

    train_index, val_index = next(
        split_train_val.split(
            df_train_val, df_train_val[ADDITIONAL_TARGET_COLUMN_CATEGORIES]
        )
    )
    df_train = df_train_val.iloc[train_index]
    df_val = df_train_val.iloc[val_index]

    df_train_final = construct_final_target_column(df=df_train)
    df_val_final = construct_final_target_column(df=df_val)
    df_test_final = construct_final_target_column(df=df_test)

    return df_train_final, df_val_final, df_test_final


def construct_final_target_column(
    *,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct final dataframe with combined categories/transaction label target_column

    :param df: pandas DataFrame after split - wide format
    :return: pandas DataFrame in final long format
    """

    # Take the additional targets for each row and rename to target column
    df_additional_label = df.drop(TARGET_COLUMN, axis=1)
    df_additional_label.rename(
        columns={ADDITIONAL_TARGET_COLUMN_CATEGORIES: TARGET_COLUMN}, inplace=True
    )

    # Take the transcaction labels if the exist and explode to long format
    df_transaction_label = df[~(df[TARGET_COLUMN].isnull())].drop(
        ADDITIONAL_TARGET_COLUMN_CATEGORIES, axis=1
    )
    df_transaction_label = df_transaction_label.explode(TARGET_COLUMN)

    # Append transaction labels rows to category rows
    df_final = pd.concat([df_additional_label, df_transaction_label], join="inner")

    # define mapping of labels to int
    label_to_int = dict(zip(LABELS, map(int, range(len(LABELS)))))

    df_final[TARGET_COLUMN_INT] = df_final[TARGET_COLUMN].map(label_to_int)

    return df_final
