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
import pickle as p
import s3fs
import tensorflow as tf

from typing import List, Any, Dict

user_cats = [
    'user_id',
    'point_of_sale',
    'geo_location_country',
    'destination_id',
    'sort_type',
    'search_timestamp_month',
    'search_timestamp_dayofweek',
    'checkin_date_month',
    'checkin_date_dayofweek',
    'checkout_date_month',
    'checkout_date_dayofweek'
]

user_numerical_columns = [
    'days_till_trip',
    'length_of_stay',
    'is_mobile',
    'adult_count',
    'child_count',
    'infant_count',
    'room_count'
]

user_multivalent_categoricals = ["applied_filters"]
item_categorical_columns = [
    'impression_prop_id',
    'impression_review_rating',
    'impression_review_count',
    'impression_star_rating',
    'impression_price_bucket']
item_numerical_columns = [
    'AirConditioning',
    'AirportTransfer',
    'Bar',
    'FreeAirportTransportation',
    'FreeBreakfast',
    'FreeParking',
    'FreeWiFi',
    'Gym',
    'HighSpeedInternet',
    'HotTub',
    'LaundryFacility',
    'Parking',
    'PetsAllowed',
    'PrivatePool',
    'SpaServices',
    'SwimmingPool',
    'WasherDryer',
    'WiFi'
]


def parse_example(serialized_example) -> Dict[str, tf.Tensor]:
    """
    reads serialized tf-records and transforms them into dict that holds the feature name and a tensor
    to be used with train and val data
    Args:
        serialized_example: tf-record example
    Returns: dict with features as keys and tensors as values
    """
    numerical = {c: tf.io.FixedLenFeature((), tf.float32) for c in user_numerical_columns}
    cats = {c: tf.io.FixedLenFeature((), tf.string) for c in user_cats}
    multivalent_categoricals = {c: tf.io.FixedLenSequenceFeature((), tf.string, allow_missing=True) for c in
                                user_multivalent_categoricals}
    item_numerical = {c: tf.io.FixedLenFeature((), tf.float32) for c in item_numerical_columns}
    item_cats = {c: tf.io.FixedLenFeature((), tf.string) for c in item_categorical_columns}

    context, sequences, _ = tf.io.parse_sequence_example(
        serialized=serialized_example,
        context_features={**numerical, **item_numerical, **item_cats, **cats},
        sequence_features=multivalent_categoricals
    )

    parsed_data = {k: v for k, v in context.items()
                   if k not in user_numerical_columns and k not in item_numerical_columns}
    parsed_data.update(sequences)
    parsed_data["user_numerical"] = tf.concat([tf.expand_dims(context[c], 1)
                                               for c in user_numerical_columns], axis=-1)
    parsed_data["item_numerical"] = tf.concat([tf.expand_dims(context[c], 1)
                                               for c in item_numerical_columns], axis=-1)

    return parsed_data


def parse_candidate_example(serialized_example) -> Dict[str, tf.Tensor]:
    """
    Like parse_example reads tf-records and spits out a dict of feature names as keys and tensors as values
    to be used with "candidate" tf-records. refer to the etl for the difference between train, val, and candidate
    tf-records
    Args:
        serialized_example: tf-record example
    Returns: dict with features as keys and tensors as values
    """
    numerical = {c: tf.io.FixedLenFeature((), tf.float32) for c in item_numerical_columns}

    cats = {c: tf.io.FixedLenFeature((), tf.string) for c in item_categorical_columns}

    context, sequences, _ = tf.io.parse_sequence_example(
        serialized=serialized_example,
        context_features={**numerical, **cats},
        sequence_features=None
    )

    parsed_data = {k: v for k, v in context.items() if k not in item_numerical_columns}
    parsed_data.update(sequences)
    parsed_data["item_numerical"] = tf.concat([tf.expand_dims(context[c], 1)
                                               for c in item_numerical_columns], axis=-1)

    return parsed_data


def read_pickle(path: str) -> Any:
    """
    Abstracts reading pickled files from s3 vs local file sys
    Args:
        path: path where pickled obj lives
    Returns: the un-pickled object
    """
    if "s3" in path:
        fs = s3fs.S3FileSystem(
            anon=False,
            s3_additional_kwargs={
                "ServerSideEncryption": "AES256"
            }
        )
        file_sys = fs.open(path, "rb")
    else:
        file_sys = open(path, "rb")

    obj = p.load(file_sys)

    return obj


def save_pickle(obj, path: str):
    """
    Simple functon that abstracts saving pickle files to s3 vs local filesys.
    Args:
        obj: python object to save using pickle
        path: path to save obj at (can be s3 or local filesys)
    """
    if len(path) >= 2 and "s3" == path[:2]:
        fs = s3fs.S3FileSystem(
            anon=False,
            s3_additional_kwargs={
                "ServerSideEncryption": "AES256"
            })
        file_sys = fs.open(path, "wb")
    else:
        file_sys = open(path, "wb")

    p.dump(obj, file_sys)


def list_tf_records(path: str) -> List[str]:
    """
    Abstracts getting tf-records for feeding into tensorflow data training pipeline
    Args:
        path: path where tf records live
    Returns: List of paths to tf-record files
    """
    # part-r is due to library used to transform parquets
    records = [os.path.join(path, file) for file in os.listdir(path) if "part-r" in file]

    if len(records) == 0:
        raise ValueError("No files were found!")

    return records
