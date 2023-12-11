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
import click
import tensorflow as tf

from eg_two_tower.training import train, DataPaths, TrainOpts
from eg_two_tower.model import ModelHyperOpts, CategoricalFeature, NumericalFeature

from loguru import logger

from pprint import pformat
from dataclasses import asdict
from typing import Optional


@click.command()
@click.option('--data_dir',
              help="Directory where the tf-records exist for train, validation, and testing exist.")
@click.option("--pickle_dir",
              help="Directory where the pickled vocab and probabilities exist. This only exists due to how accessing"
                   "different file systems in DataBricks is setup.")
@click.option("--tensorboard_log_dir",
              help="Directory to store tensorboard logs. If the directory does not exist it is created.",
              default="tensorboard-runs")
@click.option("--metrics_path",
              help="Path to save csv file of metrics",
              default="model-out/metrics.csv")
@click.option("--prob_correction", default=True)
@click.option("--numerical_input_batch_norm", default=True)
@click.option("--output_l2", default=True)
@click.option("--nb_epochs", default=50)
@click.option("--learning_rate", default=0.001)
@click.option("--batch_size", default=2048)
def main(
        data_dir: str,
        pickle_dir: Optional[str] = None,
        tensorboard_log_dir: str = "tensorboard-runs",
        metrics_path: str = "model-out/metrics.csv",
        prob_correction: bool = True,
        numerical_input_batch_norm: bool = True,
        output_l2: bool = True,
        nb_epochs: int = 50,
        learning_rate: float = 0.001,
        batch_size: int = 2048
):
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    if pickle_dir is None:
        pickle_dir = data_dir

    data_paths = DataPaths(
        train=os.path.join(data_dir, "train"),
        valid=os.path.join(data_dir, "val"),
        candidates=os.path.join(data_dir, "candidates"),
        vocab=os.path.join(pickle_dir, "vocab"),
        probabilities=os.path.join(pickle_dir, "prob"),
        tensorboard_log_dir=tensorboard_log_dir,
        metrics=metrics_path
    )
    data_paths_dict = asdict(data_paths)
    data_paths_dict_str = pformat(data_paths_dict)
    logger.info("Data paths:")
    logger.info(data_paths_dict_str)

    train_opts = TrainOpts(
        nb_epochs=nb_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        recall_true_key="impression_prop_id",
        candidate_id="impression_prop_id",
        recall_at_k=[10, 50, 100],
        prob_correction=prob_correction
    )
    train_paths_dict = asdict(train_opts)
    train_paths_dict_str = pformat(train_paths_dict)
    logger.info("Training Options:")
    logger.info(train_paths_dict_str)

    user_model_opts = ModelHyperOpts(
        layer_sizes=[64, 64],
        numerical=[
            NumericalFeature("days_till_trip"),
            NumericalFeature('length_of_stay'),
            NumericalFeature('is_mobile'),
            NumericalFeature('adult_count'),
            NumericalFeature('child_count'),
            NumericalFeature('infant_count'),
            NumericalFeature('room_count')
        ],
        categoricals=[
            CategoricalFeature(name='point_of_sale', output_dim=16),
            CategoricalFeature(name='geo_location_country', output_dim=16),
            CategoricalFeature(name='destination_id', output_dim=16),
            CategoricalFeature(name='search_timestamp_month', output_dim=16),
            CategoricalFeature(name='search_timestamp_dayofweek', output_dim=16),
            CategoricalFeature(name='checkin_date_month', output_dim=16),
            CategoricalFeature(name='checkin_date_dayofweek', output_dim=16),
            CategoricalFeature(name='checkout_date_month', output_dim=16),
            CategoricalFeature(name='checkout_date_dayofweek', output_dim=16)
        ],
        multivalent_categoricals=[CategoricalFeature(name="applied_filters", output_dim=16)],
        numerical_input_batch_norm=numerical_input_batch_norm,
        numerical_layer_size=None,
        output_l2=output_l2
    )

    item_model_opts = ModelHyperOpts(
        layer_sizes=[64, 64],
        numerical=[
            NumericalFeature('AirConditioning'),
            NumericalFeature('AirportTransfer'),
            NumericalFeature('Bar'),
            NumericalFeature('FreeAirportTransportation'),
            NumericalFeature('FreeBreakfast'),
            NumericalFeature('FreeParking'),
            NumericalFeature('FreeWiFi'),
            NumericalFeature('Gym'),
            NumericalFeature('HighSpeedInternet'),
            NumericalFeature('HotTub'),
            NumericalFeature('LaundryFacility'),
            NumericalFeature('Parking'),
            NumericalFeature('PetsAllowed'),
            NumericalFeature('PrivatePool'),
            NumericalFeature('SpaServices'),
            NumericalFeature('SwimmingPool'),
            NumericalFeature('WasherDryer'),
            NumericalFeature('WiFi')
        ],
        categoricals=[
            CategoricalFeature(name='impression_prop_id', output_dim=32),
            CategoricalFeature(name='impression_review_rating', output_dim=16),
            CategoricalFeature(name='impression_review_count', output_dim=16),
            CategoricalFeature(name='impression_star_rating', output_dim=16),
            CategoricalFeature(name='impression_price_bucket', output_dim=16)
        ],
        multivalent_categoricals=None,
        numerical_input_batch_norm=numerical_input_batch_norm,
        numerical_layer_size=None,
        output_l2=output_l2
    )

    tf.keras.backend.clear_session()
    # not adding run_functions_eagerly leads to error in newer tf versions...
    tf.config.run_functions_eagerly(True)

    train(
        data_paths=data_paths,
        train_opts=train_opts,
        user_model_opts=user_model_opts,
        item_model_opts=item_model_opts
    )


if __name__ == '__main__':
    main()
