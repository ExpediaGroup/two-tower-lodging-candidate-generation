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

import platform
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_recommenders as tfrs
from eg_two_tower.data import list_tf_records, parse_candidate_example, read_pickle, parse_example
from eg_two_tower.model import UserModel, ItemModel, TwoTowerModel, ModelHyperOpts
from dataclasses import dataclass, field

from typing import Dict, Any, List, Tuple, Callable

if platform.processor() == "arm":
    # use the legacy Adam implementation if running on M1/M2 machine (suggested by tensorflow).
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam


@dataclass
class DataPaths:
    """
    Simple dataclass for passing around data paths

    Args:
        train: path to the tf-records used for training
        valid: path to the tf-records used for validation
        candidates: path to the tf-records used for setting up the candidates in tf-recommenders
        vocab: path to the tf-records used for training
        probabilities: path to a python dict where the key is the item id and the value is the probability of it occurring
            in the training dataset.
        metrics: Where to save metrics csv file
        tensorboard_log_dir: Where to read tensorflow logs from
        final_model_weights: Where the model weights will be saved
    """
    train: str
    valid: str
    candidates: str
    vocab: str
    probabilities: str
    metrics: str
    tensorboard_log_dir: str
    final_model_weights: str = None


@dataclass
class TrainOpts:
    """
    nb_epochs: number of full passes over the data
    batch_size: number of examples per mini batch (we used 2048 but depending on hardware etc you may want
        to change this)
    learning_rate: hyperparameter for sgd (.001 the default has worked decently)
    recall_at_k: a list of ints for how many retrieved items we should consider
    recall_true_key: name of the key in the input dict to the model that contains the target ids

    candidate_id: name of the key in the input dict from the candidates tf-records dataset that contains the distinct
        ids that occurred in the dataset
    remove_accidental_hits:  If true, when given
        enables removing accidental hits of examples used as negatives. An
        accidental hit is defined as a candidate that is used as an in-batch
        negative but has the same id with the positive candidate. (from tf-recommenders)
    prob_correction: If true then optional tensors of candidate sampling
        probabilities. When given will be used to correct the logits to
        reflect the sampling probability of negative candidates. (from tf-recommenders)
    """
    nb_epochs: int
    batch_size: int
    learning_rate: float = 0.001

    recall_at_k: List[int] = field(default_factory=lambda: [10, 50, 100])
    recall_true_key: str = "impression_prop_id"

    candidate_id: str = "impression_prop_id"
    remove_accidental_hits: bool = True
    prob_correction: bool = True


def prob_correction(probs: Dict[str, float], candidate_id_key: str) -> Callable:
    """
    abstracts tensorflow operation for adding probabilities of the items in the current batch dict
    Args:
        probs: dict where the key is the item id and the value is probability of that item occurring in dataset
        candidate_id_key: name of the key that the model expects when accessing in the dict that is passed to it
    Returns: function that can be called with input dict
    """
    keys, items = [], []
    for k, v in probs.items():
        keys.append(k)
        items.append(v)

    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.convert_to_tensor(keys, dtype=tf.string),
            values=tf.convert_to_tensor(items, dtype=tf.float32)
        ),
        default_value=0
    )

    def _prob_correction(example):
        example["candidate_sampling_probability"] = tf.map_fn(
            lambda x: table.lookup(x),
            example[candidate_id_key],
            fn_output_signature=tf.float32)

        return example

    return _prob_correction


def train(
        data_paths: DataPaths,
        train_opts: TrainOpts,
        user_model_opts: ModelHyperOpts,
        item_model_opts: ModelHyperOpts
):
    """
    Setups tf datasets and two tower model for training then starts training according to
    options in the opts dataclasses passed through
    Args:
        data_paths: dataclass DataPaths that stores data paths needed to train model
            refer to DataPaths for more info
        train_opts: dataclass DataPaths that contains hyperparameters for training model, e.g., learning rate
            refet to TrainOpts for more info
        user_model_opts: dataclass ModelHyperOpts that stores hyperparameters for the user tower e.g., number of layers
            refer to ModelHyperOpts for more info
        item_model_opts: dataclass ModelHyperOpts that stores hyperparameters for the item tower e.g., number of layers
            refer to ModelHyperOpts for more info
    """
    train_dataset, valid_dataset, candidates_dataset = read_data(data_paths, train_opts)
    vocab = read_pickle(data_paths.vocab)
    probs = read_pickle(data_paths.probabilities)

    if train_opts.prob_correction:
        train_dataset = train_dataset.map(prob_correction(probs, train_opts.candidate_id))

    user_model = UserModel(
        user_model_opts.layer_sizes,
        vocab,
        numerical_features=user_model_opts.numerical,
        categorical_features=user_model_opts.categoricals,
        multivalent_categorical_features=user_model_opts.multivalent_categoricals
    )

    item_model = ItemModel(
        item_model_opts.layer_sizes,
        vocab,
        numerical_features=item_model_opts.numerical,
        categorical_features=item_model_opts.categoricals,
        multivalent_categorical_features=item_model_opts.multivalent_categoricals)

    model = TwoTowerModel(
        user_model=user_model,
        item_model=item_model,
        vocab=vocab,
        candidate_id=train_opts.candidate_id,
        task=tfrs.tasks.Retrieval(remove_accidental_hits=train_opts.remove_accidental_hits))

    optimizer = Adam(learning_rate=train_opts.learning_rate)
    model.compile(optimizer=optimizer)

    # run one minibatch through model to setup tensor sizes...
    for x in train_dataset:
        model(x)
        break

    scann_params = dict(
        k=100,
        num_reordering_candidates=500,
        num_leaves_to_search=30
    )

    model, metrics = train_loop(
        model,
        scann_params,
        train_dataset,
        valid_dataset,
        candidates_dataset,
        train_opts,
        data_paths)

    if data_paths.final_model_weights is not None:
        model.save_weights(data_paths.final_model_weights)

    if data_paths.metrics is not None:
        metrics.to_csv(data_paths.metrics, index=False)


def read_data(
        data_paths: DataPaths,
        train_opts: TrainOpts,
        num_parallel_reads: int = 8,
        buffer_size: int = 10_000
) -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
    """
    abstracts away reading and setting up tf-records pipelines for training
    Args:
        data_paths: a DataPaths dataclass storing paths to train, valid, and candidate tf-records
            refer to DataPaths for more info
        train_opts: a TrainOpts dataclass storing hyperparameters for training, e.g., learning rate
            refer to TrainOpts for more info
        num_parallel_reads: A `tf.int64` scalar representing the
            number of files to read in parallel. If greater than one, the records of
            files read in parallel are outputted in an interleaved order. If your
            input pipeline is I/O bottlenecked, consider setting this parameter to a
            value greater than one to parallelize the I/O. If `None`, files will be
            read sequentially. (from tf.data.TFRecordDataset)

            8 Seemed to work fine...
        buffer_size: A `tf.int64` scalar representing the number of
            bytes in the read buffer. If your input pipeline is I/O bottlenecked,
            consider setting this parameter to a value 1-100 MBs. If `None`, a
            sensible default for both local and remote file systems is used.
            (from tf.data.TFRecordDataset)

            10_000 seemed to work fine
    Returns: Three tf.data.TFRecordDataset for train, valid, and candidates.

    """
    candidates_files = list_tf_records(data_paths.candidates)

    candidates_dataset = (
        tf.data.TFRecordDataset(candidates_files, num_parallel_reads=num_parallel_reads, compression_type="GZIP")
        .batch(train_opts.batch_size)
        .map(parse_candidate_example, num_parallel_calls=num_parallel_reads))

    train_files = list_tf_records(data_paths.train)

    train_dataset = (
        tf.data.TFRecordDataset(train_files, num_parallel_reads=num_parallel_reads, compression_type="GZIP")
        .shuffle(buffer_size, reshuffle_each_iteration=True)
        .batch(train_opts.batch_size)
        .map(parse_example, num_parallel_calls=num_parallel_reads)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid_files = list_tf_records(data_paths.valid)

    valid_dataset = (
        tf.data.TFRecordDataset(valid_files, num_parallel_reads=num_parallel_reads, compression_type="GZIP")
        .batch(train_opts.batch_size)
        .map(parse_example, num_parallel_calls=num_parallel_reads)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, valid_dataset, candidates_dataset


def train_loop(
        model: TwoTowerModel,
        scann_params: Dict[str, Any],
        train_dataset: tf.data.TFRecordDataset,
        valid_dataset: tf.data.TFRecordDataset,
        candidates_dataset: tf.data.TFRecordDataset,
        train_opts: TrainOpts,
        data_paths: DataPaths
) -> Tuple[TwoTowerModel, pd.DataFrame]:
    """
    Mainly to be called by the function train (defined above), this fucntion  does the actual "training" while train
        sets everything up for train_loop then calls it.
    Args:
        model: a setup TwoTowerModel (usually done in train)
        scann_params: kwargs for the tfrs.tasks.Retrieval class
            refer to the class here https://www.tensorflow.org/recommenders/api_docs/python/tfrs/tasks/Retrieval
        train_dataset: TFRecordDataset for the training files (usually setup in read_data)
        valid_dataset: TFRecordDataset for the validation files (usually setup in read_data)
        candidates_dataset: TFRecordDataset for the candidate files (usually setup in read_data)
        train_opts: a TrainOpts dataclass storing hyperparameters for training, e.g., learning rate
            refer to TrainOpts for more info
        data_paths: a DataPaths dataclass storing paths to train, valid, and candidate tf-records
            refer to DataPaths for more info

    Returns: The trained model and a pandas DataFrame with the computed metrics across epochs
    """
    summary_writer = tf.summary.create_file_writer(data_paths.tensorboard_log_dir)
    compute_write_recall = ComputeWriteRecall(train_opts, summary_writer)

    for epoch in range(1, train_opts.nb_epochs + 1):
        history = model.fit(
            train_dataset,
            epochs=1,
            callbacks=[tf.keras.callbacks.TerminateOnNaN()]
        )
        history = {k: v[0] for k, v in history.history.items()}
        write_metrics(summary_writer, epoch - 1, history)

        scann = tfrs.layers.factorized_top_k.ScaNN(**scann_params)
        scann.index_from_dataset(
            tf.data.Dataset.zip(candidates_dataset.map(lambda d: (d[model.candidate_id], model.item_model_call(d))))
        )

        model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(candidates=scann)
        optimizer_config = model.optimizer.get_config()
        optimizer = Adam.from_config(optimizer_config)
        compute_write_recall(model, scann, valid_dataset)
        model.compile()
        results = model.evaluate(valid_dataset, return_dict=True, verbose=0)
        update_metrics(compute_write_recall.training_metrics, results)
        write_metrics(summary_writer, epoch - 1, results)

        metric = tfrs.metrics.FactorizedTopK(
            candidates=tf.data.Dataset.zip(
                candidates_dataset.map(lambda d: (d[model.candidate_id], model.item_model_call(d)))))
        model.task = tfrs.tasks.Retrieval(
            metrics=metric,
            remove_accidental_hits=train_opts.remove_accidental_hits)

        model.compile(optimizer=optimizer)

    training_metrics = pd.DataFrame(compute_write_recall.training_metrics)
    training_metrics["epoch"] = list(range(1, train_opts.nb_epochs + 1))

    return model, training_metrics


class ComputeWriteRecall:
    """
    Simple helper class that abstracts away model and SCANN for ann search and keeps track of recall during training
    """

    def __init__(
            self,
            train_opts: TrainOpts,
            summary_writer: tf.summary.SummaryWriter
    ):
        """
        Args:
            train_opts: the same TrainOpts being used during training
            summary_writer: a tensorflow SummaryWriter for writing to tensorboard
        """
        super(ComputeWriteRecall, self).__init__()

        self.recall_at_k: List[int] = train_opts.recall_at_k
        self.recall_true_key: str = train_opts.recall_true_key
        self.summary_writer = summary_writer
        self.training_metrics = {}
        self.epoch = 1

    def __call__(
            self,
            model: TwoTowerModel,
            scann: tfrs.layers.factorized_top_k.ScaNN,
            valid_dataset: tf.data.TFRecordDataset,
    ):
        """
        Called during training (in the train_loop function) to compute and write recall
        Args:
            model: TwoTower model during training
            scann: the SCANN object being used
            valid_dataset: the tf-records dataset
        """
        for k in self.recall_at_k:
            ground_truth, approx_results = [], []

            for batch in valid_dataset:
                scores, ids = scann(model.user_model_call(batch), k=k)
                approx_results.append(ids)
                ground_truth.append(batch[self.recall_true_key])

            recall = compute_recall(ground_truth, approx_results)
            recall_metrics = {f"recall@{k}": recall}

            update_metrics(self.training_metrics, recall_metrics)
            write_metrics(self.summary_writer, self.epoch, recall_metrics)

        self.epoch += 1


def update_metrics(running_metrics: Dict[str, List[float]], step_metrics: Dict[str, float]):
    """
    Used with ComputeWriteRecall and within train_loop to update a dict that holds results for different
        metric calculations.
    Args:
        running_metrics:
        step_metrics:
    """
    for k, v in step_metrics.items():
        if k in running_metrics:
            running_metrics[k].append(v)
        else:
            running_metrics[k] = [v]


def write_metrics(
        summary_writer: tf.summary.SummaryWriter,
        step: int,
        metrics: Dict[str, float]
):
    """
    Abstracts writing dict of metrics for tensorboard
    Args:
        summary_writer: the SummaryWriter being used throughout the training process
        step: some value for index across time in this case epoch is teh default but can technically be training
            step (i.e., metrics from minibatch in sgd)
        metrics: a dict storing the results for different metrics at step key is the metric name and the value is well
            the value of for that metric...
    """
    with summary_writer.as_default():
        for k, v in metrics.items():
            tf.summary.scalar(k, v, step)


def compute_recall(
        ground_truth: List[List[int]],
        approx_results: List[List]
) -> float:
    """
    Computes recall for the given ground_truth and predictions (approx_results) given a list of lists that contain
        different queries (examples).
    NOTE that this does not compute recall@k explicitly if you want to compute recall@k just pass in approx_results
        up to k.
    Args:
        ground_truth: list of the items we are trying to predict
        approx_results: predicted ids (in this code it's assumed to be from scann)
    Returns: average over all recall calculations
    """
    return np.mean([
        len(np.intersect1d(truth, approx)) / len(truth)
        for truth, approx in zip(ground_truth, approx_results)
    ])
