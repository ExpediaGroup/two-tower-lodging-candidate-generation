"""
Copyright [2023] Expedia, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import numpy as np
import pyspark.sql.functions as F

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

from typing import List, Tuple, Dict

from eg_two_tower.data import save_pickle
from eg_two_tower.etl_defaults import DEFAULT_VALIDATION_DATE_SPLIT, IMPRESSION_COLUMNS, FINAL_AMENITY_COLUMNS, \
    CANDIDATE_IMPRESSION_COLUMNS, QUERY_CATEGORICAL_COLUMNS, IMPRESSION_CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, \
    FINAL_COLUMNS

from loguru import logger


def run(
        spark: SparkSession,
        main_tsv: str,
        amenities_tsv: str,
        out_dir: str,
):
    """
    Runs the whole ETL that is spark operations and then saving as tf-records alongside pickled metadata.

    Args:
        spark: An initialized spark session. This spark session is passed in for easier use with DataBricks env.
        main_tsv: Location of main.tsv
        amenities_tsv: location of amenities.tsv
        out_dir: dir to save output of etl
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    prob_out_path = os.path.join(out_dir, "prob")
    candidates_out_path = os.path.join(out_dir, "candidates")
    train_out_path = os.path.join(out_dir, "train")
    val_out_path = os.path.join(out_dir, "val")
    vocab_out_path = os.path.join(out_dir, "vocab")

    main_tsv = spark.read.csv(main_tsv, sep="\t", header=True, inferSchema=True)
    amenities = spark.read.csv(amenities_tsv, sep="\t", header=True, inferSchema=True)

    main_processed = format_impressions(main_tsv, amenities, IMPRESSION_COLUMNS)
    # noinspection PyTypeChecker
    main_processed_clicks_only = main_processed.filter(F.col("clicked") == 1)

    candidates = setup_candidates(main_processed, CANDIDATE_IMPRESSION_COLUMNS, FINAL_AMENITY_COLUMNS,
                                  IMPRESSION_CATEGORICAL_COLUMNS, FINAL_AMENITY_COLUMNS)
    write_tf_records(candidates, candidates_out_path)
    logger.info(f"wrote candidates tf-records to: {candidates_out_path}")
    train, val = validation_split(main_processed_clicks_only, DEFAULT_VALIDATION_DATE_SPLIT)

    train = setup_train(train, FINAL_COLUMNS, IMPRESSION_CATEGORICAL_COLUMNS + QUERY_CATEGORICAL_COLUMNS,
                        NUMERICAL_COLUMNS)
    write_tf_records(train, train_out_path)
    logger.info(f"wrote train tf-records to: {train_out_path}")

    val = setup_validation(val, FINAL_COLUMNS, IMPRESSION_CATEGORICAL_COLUMNS + QUERY_CATEGORICAL_COLUMNS,
                           NUMERICAL_COLUMNS)
    write_tf_records(val, val_out_path)
    logger.info(f"wrote val tf-records to: {val_out_path}")

    vocab = build_vocab(train, IMPRESSION_CATEGORICAL_COLUMNS + QUERY_CATEGORICAL_COLUMNS)
    save_candidate_sampling_prob(train, "impression_prop_id", prob_out_path)
    logger.info(f"Pickled candidate probabilities to: {prob_out_path}")

    save_pickle(vocab, vocab_out_path)
    logger.info(f"Pickled vocab to: {vocab_out_path}")


def format_impressions(
        data: DataFrame,
        amenities: DataFrame,
        impression_columns: List[Tuple[str, str]]
) -> DataFrame:
    """
    Takes in a spark DataFrame (main.tsv) and extracts the impressions that are stored as lists of strings where each
        string is contains pipe delimited information about the impression.
    Args:
        data: Spark DataFrame from the file main.tsv
        amenities: Spark DataFrame from the file amenities.tsv
        impression_columns: A list of tuples where the fist element is a string representing one of the columns in
            the pipe delimited impression string and the second element is a string for the data type
            e.g., string, float.
    Returns:
    """
    main_processed = (
        data
        .filter(F.col("checkin_date").isNotNull())
        .withColumn("event_date", F.to_date("search_timestamp"))
        .withColumn("impressions", F.split('impressions', "\|"))
        .withColumn("impression", F.explode("impressions"))
        .withColumn("impression", F.split("impression", ","))
        .withColumn("applied_filters", F.split("applied_filters", "\|")).alias("applied_filters"))

    main_processed = _set_impression_columns(main_processed, impression_columns)
    main_processed = add_features(main_processed, amenities)

    return main_processed


def write_tf_records(data: DataFrame, path: str):
    """
    Helper function for writing tf-records (just to save some space...)
    Args:
        data: a spark DataFrame to be written as tf-records
        path: path to write to
    """

    dtype_map = {"timestamp": "string"}

    for column, dtype in data.dtypes:
        if dtype in dtype_map:
            data = data.withColumn(column, F.col(column).astype(dtype_map[dtype]))
    data \
        .write \
        .mode("overwrite") \
        .format("tfrecords") \
        .option("recordType", "SequenceExample") \
        .option("codec", "org.apache.hadoop.io.compress.GzipCodec") \
        .save(path)


def validation_split(data: DataFrame, date: str) -> Tuple[DataFrame, DataFrame]:
    """
    Simple function for splitting a spark DataFrame of data into train and val.
    Args:
        data: A spark DataFrame that has an event_date column
        date: A string with format YYYY-MM-DD
    Returns: A tuple of dataframes
    """
    # noinspection PyTypeChecker
    train = data.filter(F.col("event_date") < date)
    # noinspection PyTypeChecker
    val = data.filter(F.col("event_date") >= date)

    return train, val


def _set_impression_columns(data: DataFrame, impression_columns: List[Tuple[str, str]]) -> DataFrame:
    """
    Helper function for format_impressions. This function extracts impression features from the impression column
    and makes them columns in data.

    Args:
        data: Data derived from the RecTour dataset that has had the impressions extracted (refer to format_impressions
            for more details).
        impression_columns: A list of tuples where the first element in the tuple is a string for the column in the
            data and the second element is the column type.
    Returns: data with new columns from impression_columns
    """
    for i, (c, col_type) in enumerate(impression_columns):
        data = data.withColumn(f"impression_{c}", F.col("impression").getItem(i).astype(col_type))

    return data


def setup_candidates(
        data: DataFrame,
        candidate_impression_columns: List[str],
        amenities_columns: List[str],
        categorical_columns: List[str],
        numerical_columns: List[str]
) -> DataFrame:
    """
    Generates a spark DataFrame with the unique item ids and their corresponding features that are defined by
        amenities_columns, categorical_columns, numerical_columns
    Args:
        data: A spark DataFrame (should be the data from main.tsv)
        candidate_impression_columns: The column in the data to use to collect the unique item ids.
        amenities_columns: Column that holds amenities
        categorical_columns: Columns to collect that represent categorical features for the item
        numerical_columns: Columns to collect that represent numerical features for the item

    Returns: A spark DataFrame that contains all unique item ids and their respective categorical and numerical
        features
    """
    candidates = data.select("impression_prop_id", *candidate_impression_columns, *amenities_columns).distinct()
    candidates = candidates.groupBy("impression_prop_id").agg(
        *[F.ceil(F.avg(c)).alias(c) for c in candidate_impression_columns + amenities_columns])
    candidates = map_fill_categorical_columns(candidates, categorical_columns)
    candidates = map_numericals(candidates, numerical_columns)

    return candidates


def setup_train(train: DataFrame, final_columns: List[str], categorical_columns: List[str],
                numerical_columns: List[str]) -> DataFrame:
    """
    Preps training data by imputing missing categorical and makes sure that numerical features are
        defined as floats.
    Args:
        train: A spark DataFrame representing the training data extracted from main.tsv
        final_columns: List of columns to keep
        categorical_columns: Columns representing categorical features to impute and map to ids
        numerical_columns: List of numerical columns used for training that will be cast to floats.

    Returns: The mapped training data as a spark DataFrame
    """
    train = _setup(train, final_columns, categorical_columns, numerical_columns)

    return train


def setup_validation(val: DataFrame, final_columns: List[str], categorical_columns: List[str],
                     numerical_columns: List[str]) -> DataFrame:
    """
    Preps validation data by imputing missing categorical and makes sure that numerical features are
        defined as floats.
    Args:
        val: A spark DataFrame representing the validation data extracted from main.tsv (usually resulting from the
            validation_split function in this file).
        final_columns: List of columns to keep
        categorical_columns: Columns representing categorical features to impute and cast to strings
        numerical_columns: List of numerical columns used for training that will be cast to floats.

    Returns: The mapped training data as a spark DataFrame
    """
    val = val.withColumn(
        "clicked_impression_prop_ids",
        F.collect_list("impression_prop_id").over(
            Window.partitionBy("search_id").orderBy(F.desc("impression_is_trans"),
                                                    F.desc("impression_num_clicks"))))
    val = val.withColumn(
        "rank",
        F.row_number().over(Window.partitionBy("search_id").orderBy(F.desc("search_timestamp"))))
    # noinspection PyTypeChecker
    val = val.filter(F.col("rank") == 1)
    val = _setup(val, final_columns + ["clicked_impression_prop_ids"], categorical_columns, numerical_columns)

    return val


def _setup(data: DataFrame, final_columns: List[str], categorical_columns: List[str],
           numerical_columns: List[str]) -> DataFrame:
    """
    Helper function or train and valid split versions of this function.
    Preps training data by imputing missing categorical and makes sure that numerical features are
        defined as floats.
    Args:
        data: A spark DataFrame representing some data extracted from main.tsv
        final_columns: List of columns to keep
        categorical_columns: Columns representing categorical features to impute and cast to strings
        numerical_columns: List of numerical columns used for training that will be cast to floats.
    Returns: The mapped training data as a spark DataFrame
    """
    data = data.select(final_columns)
    data = map_fill_categorical_columns(data, categorical_columns)
    data = map_numericals(data, numerical_columns)

    return data


def map_fill_categorical_columns(
        data: DataFrame,
        categorical_columns: List[str],
        missing_token: str = "MISSING"
) -> DataFrame:
    """
    Takes a spark DataFrame and casts the columns defined as categorical as strings and imputes missing values with
        missing_token.
    Args:
        data: The spark dataframe to operate on... (should come from main.tsv)
        categorical_columns: List of columns that should be treated as categorical
        missing_token: Impute value
    Returns: data with categorical_columns cast to strings and imputed with missing_token
    """
    for c in categorical_columns:
        data = data.withColumn(c, F.col(c).astype("string"))
        data = data.withColumn(c, F.when(F.col(c).isNull(), missing_token).otherwise(F.col(c)))

    return data


def map_numericals(data: DataFrame, numerical_columns: List[str]) -> DataFrame:
    """
    Casts every column passed to this function to float types for use in tensorflow.
    Args:
        data: spark DataFrame to operate on (should come from main.tsv)
        numerical_columns: List of strings that represent the columns to cast to float
    Returns: data with numerical_columns cast to float
    """
    for c in numerical_columns:
        data = data.withColumn(c, F.col(c).astype("float"))

    return data


def add_features(data: DataFrame, amenities: DataFrame) -> DataFrame:
    """
    Adds features to the data being passed by calling the helped functions associated with different type of features.
    Args:
        data: data to add features to... (should come from main.tsv)
        amenities: a spark DataFrame containing amenities data (should be from amenities.tsv)
    Returns:
    """
    data = join_amenities(data, amenities)
    data = add_impression_interaction_features(data)
    data = add_date_features(data)
    data = add_trip_features(data)

    return data


def join_amenities(data: DataFrame, amenities: DataFrame) -> DataFrame:
    """
    Helper function for joining amenities (amenities.tsv) to impression data (main.tsv)
    Args:
        data: (should come from main.tsv)
        amenities: (should come from amenities.tsv)
    Returns: Impression data joined with the amenities data
    """
    data = data.join(amenities, F.col("impression_prop_id") == F.col("prop_id"), "left")
    data = data.drop("prop_id")

    for c in amenities.columns[1:]:
        data = data.withColumn(c, F.when(F.col(c).isNull(), 0).otherwise(F.col(c)))

    return data


def add_impression_interaction_features(data: DataFrame) -> DataFrame:
    """
    Adds a binary valued column named clicked that represents whether the user clicked on the impression they saw.
    Args:
        data: (should come from main.tsv)
    Returns: data DataFrame with a new column "clicked"

    """
    # noinspection PyTypeChecker
    data = data.withColumn("clicked", F.when(F.col('impression_num_clicks') > 0, 1).otherwise(0))

    return data


def add_date_features(data: DataFrame) -> DataFrame:
    """
    Adds date features using predefined datetime columns
    Args:
        data: (should come from main.tsv)
    Returns: data with month and dayofweek features for the columns defined below

    """
    columns = ["search_timestamp", "checkin_date", "checkout_date"]

    for c in columns:
        data = data.withColumn(f"{c}_month", F.month(c))
        data = data.withColumn(f"{c}_dayofweek", F.dayofweek(c))

    return data


def add_trip_features(data: DataFrame) -> DataFrame:
    """
    Generates features from the trip context in the impression data.
    Args:
        data: (should come from main.tsv)
    Returns: data spark DataFrame with trip features
    """
    data = data.withColumn("days_till_trip", F.datediff(F.col("checkout_date"), F.col("event_date")))
    data = data.withColumn("length_of_stay", F.datediff(F.col("checkout_date"), F.col("checkin_date")))

    return data


def save_candidate_sampling_prob(data: DataFrame, candidate_column: str, path: str):
    """
    Does a simple calculation of how likely it is to see every unique item in the dataset
    Args:
        data: (should come from main.tsv)
        candidate_column: column that stores the property id
        path: path to save pickled dict to
    """
    total_count: int = data.count()
    counts: DataFrame = data.groupBy(candidate_column).count()

    prob: List[Dict[str, float]] = (
        counts
        .withColumn("sampling_prob", F.col("count") / F.lit(total_count))
        .select(candidate_column, "sampling_prob")
        .toPandas()
        .to_dict(orient="records"))

    prob: Dict[str, float] = {d[candidate_column]: d["sampling_prob"] for d in prob}

    save_pickle(prob, path)


def build_vocab(data: DataFrame, categorical_columns: List[str]) -> Dict[str, np.ndarray]:
    """
    Generates a dict where the key is the name of the column and the value is a numpy array of
        strings that represent unique values.
    Args:
        data: A spark DataFrame with columns that represent categorical features (should be data from main.tsv)
        categorical_columns: List of strings to use for building the vocab dict
    Returns: A dict with the column as a key and unique values as the dict value
    """
    data_dtypes: Dict[str, str] = dict(data.dtypes)
    vocab: Dict[str, np.ndarray] = {}

    for column in categorical_columns:
        if "array<array" in data_dtypes[column]:
            column_data = data.select(F.explode(F.flatten(column)).alias(column))
        elif "array" in data_dtypes[column]:
            column_data = data.select(F.explode(column).alias(column))
        elif "string" == data_dtypes[column]:
            column_data = data.select(F.explode(F.split(column, " ")).alias(column))
        else:
            column_data = data.select(column)

        column_data = column_data.distinct()
        # noinspection PyTypeChecker
        column_data = column_data.filter(F.col(column) != "")
        column_data = column_data.orderBy(column)
        column_data = column_data.toPandas()
        vocab[column] = np.squeeze(column_data.values)

    return vocab


def get_null_counts(data: DataFrame) -> DataFrame:
    """
    Aggregates number of nulls for each column in DataFrame
    Args:
        data: data to calculate nulls % over
    Returns: DataFrame with the same columns as the input data but as percents of null
    """
    allowed_dtypes = ["long", "integer", "float", "double"]
    null_counts = data.select(
        [(F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)) / F.count("*")).alias(c) if dtype in allowed_dtypes else
         (F.count(F.when(F.col(c).isNull(), c)) / F.count("*")).alias(c)
         for c, dtype in data.dtypes]
    )

    return null_counts
