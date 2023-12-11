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

import tensorflow as tf
import tensorflow_recommenders as tfrs
import numpy as np

from typing import Dict, Text, Tuple, List, Optional

from dataclasses import dataclass


@dataclass
class NumericalFeature:
    """
    mostly useless...mostly for compatibility with CategoricalFeature
    """
    name: str


@dataclass
class CategoricalFeature:
    """
    Used for keeping track of information for each categorical feature
    NOTE: that any feature passed to this class will be automatically be mapped to an embedding table
    if you want to use one hot or binary variables use numerical feature instead

    name: name of the feature in the input dict to model e.g., impression_price_bucket
    output_dim:
    mask_zero:
    shared_embedding_key:
    """
    name: str
    output_dim: int
    mask_zero: bool = True
    shared_embedding_key: str = None


@dataclass
class ModelHyperOpts:
    """
    layer_sizes: a list of ints that represents the size of each later. dense layers will be created for each
        new element in list.
    numerical: list of NumericalFeature dataclasses for what numerical features should be used
    categoricals: list of CategoricalFeature dataclasses for what numerical features should be used
    multivalent_categoricals: list of CategoricalFeature dataclasses for what numerical features should be used
        NOTE: these features are expected as list of entities not a single entity, e.g., amenities could fall under
        this
    output_l2: Whether to add a L2NormLayer at the end of the tower which is just a simple l2 norm applied to output
        representation of the network.
        NOTE: the default is set to True as this should improve performance
    numerical_input_batch_norm: whether to apply a batch norm to the numerical features
    numerical_layer_size: if a non None value is passed then a dense layer is created and numerical features are first
        fed through the newly created layer.
    """
    layer_sizes: Optional[List[int]] = None
    numerical: Optional[List[NumericalFeature]] = None
    categoricals: Optional[List[CategoricalFeature]] = None
    multivalent_categoricals: Optional[List[CategoricalFeature]] = None
    output_l2: bool = True
    numerical_input_batch_norm: bool = True
    numerical_layer_size: Optional[int] = None


class L2NormLayer(tf.keras.layers.Layer):
    """
    Simple function that wraps l2 norm transformation in keras layer
    """

    def __init__(self, **kwargs):
        super(L2NormLayer, self).__init__(**kwargs)

    @tf.function
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = tf.ragged.boolean_mask(inputs, mask).to_tensor()

        return tf.math.l2_normalize(inputs, axis=-1) + tf.keras.backend.epsilon()

    def compute_mask(self, inputs, mask=None):
        return mask


class TowerModel(tf.keras.Model):
    """
    Base tower keras model class for constructing "two towers"
    """

    def __init__(
            self,
            layer_sizes: List[int],
            vocab: Dict[str, Dict[str, np.ndarray]],
            numerical_features: Optional[List[NumericalFeature]] = None,
            categorical_features: Optional[List[CategoricalFeature]] = None,
            multivalent_categorical_features: Optional[List[CategoricalFeature]] = None,
            output_l2: bool = True,
            numerical_key: Optional[str] = None,
            numerical_input_batch_norm: bool = True,
            numerical_layer_size: int = None
    ):
        """
        NOTE: a lot of these are just copied up above from ModelHyperOpts...

        Args:
            layer_sizes: a list of ints that represents the size of each later. dense layers will be created for each
                new element in list.
            vocab:
            numerical_features: list of NumericalFeature dataclasses for what numerical features should be used
            categorical_features: list of CategoricalFeature dataclasses for what numerical features should be used
            multivalent_categorical_features: list of CategoricalFeature dataclasses for what numerical features should
                be used.
                NOTE: these features are expected as list of entities not a single entity, e.g., amenities could fall
                    under
            output_l2: Whether to add a L2NormLayer at the end of the tower which is just a simple l2 norm applied to output
                representation of the network.
                NOTE: the default is set to True as this should improve performance
            numerical_key: Name of the key in the input dictionary that contains the numerical features. The numerical
                features are concatenated into a vector in the tf-data pipeline
            numerical_input_batch_norm: whether to apply a batch norm to the numerical features
            numerical_layer_size: if a non None value is passed then a dense layer is created and numerical features
                will be fed through the newly created layer.

                please look at _set_numerical_nn for more details on how this happens
        """

        super().__init__()

        if all(map(lambda x: x is None, [numerical_features, categorical_features, multivalent_categorical_features])):
            raise ValueError("Some features must be set!")

        self.numerical = numerical_features
        self.categorical_features = categorical_features
        self.multivalent_categorical_features = multivalent_categorical_features
        self.output_l2 = output_l2
        self.numerical_key = numerical_key

        self.nb_numerical_features = len(self.numerical) if self.numerical is not None else None
        self.embeddings = {}
        self.numerical_encoder = None

        self.numerical_input_batch_norm = numerical_input_batch_norm
        self.numerical_layer_size = numerical_layer_size

        self.dense_layers = tf.keras.Sequential()

        self._set_model_layers(vocab, layer_sizes)

    def _set_model_layers(self, vocab: Dict[str, Dict[str, np.ndarray]], layer_sizes: List[int]):
        """
        Helper method for setting up model layers that calls other helper methods for setting up
            other parts of the network.
        Args:
            vocab: The same vocab generated from the ETL. Vocab should be a pickled dict with the name of the features
                as the keys and the distinct values that occurred in the dataset as the dict values.
            layer_sizes: The number of layers to use in the tower and their respective size
        """
        self._set_embeddings(vocab)

        if self.numerical is not None:
            self._set_numerical_nn(self.numerical_input_batch_norm, layer_size=self.numerical_layer_size)

        self._set_tower(layer_sizes)

    def _set_embeddings(self, vocab: Dict[str, Dict[str, np.ndarray]]):
        """
        One of the helper methods that is called by _set_model_layers. This method abstracts away having to manually
            setup of every feature with an embedding table (which in this case is all categorical feature for now)
            and the difference between multivalent categorical features .
        Args:
            vocab: The same vocab generated from the ETL. Vocab should be a pickled dict with the name of the features
                as the keys and the distinct values that occurred in the dataset as the dict values.
        """
        if self.categorical_features is not None:
            for feature in self.categorical_features:
                self.embeddings[feature.name] = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(vocabulary=vocab[feature.name]),
                    tf.keras.layers.Embedding(
                        input_dim=len(vocab[feature.name]) + 1,
                        output_dim=feature.output_dim,
                        mask_zero=feature.mask_zero)
                ])

        if self.multivalent_categorical_features is not None:
            for feature in self.multivalent_categorical_features:
                self.embeddings[feature.name] = tf.keras.Sequential([
                    tf.keras.layers.StringLookup(vocabulary=vocab[feature.name]),
                    tf.keras.layers.Embedding(
                        input_dim=len(vocab[feature.name]) + 1,
                        output_dim=feature.output_dim, mask_zero=feature.mask_zero),
                    tf.keras.layers.Flatten()
                ])

    def _set_numerical_nn(self, input_batch_norm: bool, layer_size: Optional[int] = None):
        """
        Another helper method called by _set_model_layers that sets up the part of the network that deals with
            numerical features and gives an option for adding a dense layer to the numerical features before they
            are concatenated with the categorical features.
        Args:
            input_batch_norm: if true adds a batch norm to the numerical features
            layer_size: hidden layer size of the optional dense layer
        """
        self.numerical_encoder = tf.keras.Sequential()

        if input_batch_norm:
            self.numerical_encoder.add(tf.keras.layers.BatchNormalization())

        if layer_size is not None:
            self.numerical_encoder.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        if layer_size is None and not input_batch_norm:
            # when no numerical features are being used
            self.numerical_encoder = None

    def _set_tower(self, layer_sizes: List[int]):
        """
        Final helper method for setting up the feedforward layers of the tower.
        Args:
            layer_sizes: a list of ints that represents how many dense layers the tower should have and their size.
        """
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.Sequential([
                tf.keras.layers.Dense(layer_size, activation="relu")
            ]))

        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

        if self.output_l2:
            self.dense_layers.add(L2NormLayer())

    def call(self, x: Dict[str, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """
        Computes and returns the representation for the given features. This is called when the object is called, e.g.,
            model = TowerModel()
            model()
        Args:
            x: a dict where the keys are either self.numerical_key or the feature names of the categorical features.
        Returns: the computed tensor
        """
        features = []

        if self.numerical is not None:
            x_numerical = self.encode_numericals(x[self.numerical_key])
            features.append(x_numerical)

        if self.categorical_features is not None:
            x_categoricals = self.encode_categorical_features(x)
            features.append(tf.keras.layers.Concatenate()(x_categoricals))

        if self.multivalent_categorical_features is not None:
            x_multivalent_categoricals = self.encode_multivalent_categoricals_features(x)
            features.append(tf.keras.layers.Concatenate()(x_multivalent_categoricals))

        x = tf.keras.layers.Concatenate()(features) if len(features) > 1 else features[0]

        return self.dense_layers(x)

    def encode_numericals(
            self,
            x: tf.Tensor
    ) -> tf.Tensor:
        """
        Helper method for computing the numerical feature's representation.
        Args:
            x: a dict where the keys are either self.numerical_key or the feature names of the categorical features.
                should be the same as what's passed in call()
        Returns: a tensor for the numerical features representation
        """
        return self.numerical_encoder(x) if self.numerical_encoder is not None else x

    def encode_categorical_features(self, x: tf.Tensor, *args, **kwargs) -> List[tf.Tensor]:
        """
        Helper method for computing the categorical feature's representation.
        Args:
            x:
        Returns:  A list of tensors for all the categorical features these can be then combined through
            concat etc
        """
        x_categoricals = []
        for feature in self.categorical_features:
            x_categorical = self.embeddings[feature.name](x[feature.name])
            x_categorical = tf.keras.layers.Flatten()(x_categorical)
            x_categoricals.append(x_categorical)
        return x_categoricals

    def encode_multivalent_categoricals_features(self, x: tf.Tensor, *args, **kwargs) -> List[tf.Tensor]:
        """
        Helper method for computing the multivalent categorical feature's representation.
        Args:
            x:
        Returns:  A list of tensors for all the multivalent categorical features these can be then combined through
            concat etc
        """
        x_multivalent_categoricals = []
        for feature in self.multivalent_categorical_features:
            x_multivalent_categorical = self.embeddings[feature.name](x[feature.name])
            x_multivalent_categoricals.append(x_multivalent_categorical)
        return x_multivalent_categoricals

    def set_embedding_matrix(self, name: str, embedding_matrix: np.ndarray):
        """
        Not used in current setup but can be used to set weights of embedding table.
        Args:
            name: name of the embedding table to set weights for
            embedding_matrix: a numpy matrix of the same size as the weight matrix of the embedding table
        """
        self.embeddings[name].layers[1].set_weights([embedding_matrix])


class ItemModel(TowerModel):
    """
    Probably unnecessary but is specific item tower implementation for later modification...
    """

    def __init__(self, layer_sizes, vocab: Dict[str, Dict[str, np.ndarray]],
                 numerical_features: List[NumericalFeature],
                 categorical_features: List[CategoricalFeature],
                 multivalent_categorical_features: List[CategoricalFeature]):
        super().__init__(layer_sizes, vocab, numerical_features, categorical_features,
                         multivalent_categorical_features, numerical_key="item_numerical")


class UserModel(TowerModel):
    """
    Not as useless as ItemModel...this is a specific implementation for the user tower
    """

    def __init__(self, layer_sizes, vocab: Dict[str, Dict[str, np.ndarray]],
                 numerical_features: List[NumericalFeature],
                 categorical_features: List[CategoricalFeature],
                 multivalent_categorical_features: List[CategoricalFeature]):
        super().__init__(layer_sizes, vocab, numerical_features, categorical_features,
                         multivalent_categorical_features, numerical_key="user_numerical")

    def call(self, x: Dict[str, tf.Tensor], *args, **kwargs) -> Dict[str, tf.Tensor]:
        """
        Refer to parent class call method
        """
        features = []

        if self.numerical is not None:
            x_numerical = self.encode_numericals(x[self.numerical_key])
            features.append(x_numerical)

        if self.categorical_features is not None:
            x_categoricals = self.encode_categorical_features(x)
            features.append(tf.keras.layers.Concatenate()(x_categoricals))

        if self.multivalent_categorical_features is not None:
            x_multivalent_categoricals = self.encode_multivalent_categoricals_features(x)
            features.append(tf.keras.layers.Concatenate()(x_multivalent_categoricals))

        x = tf.keras.layers.Concatenate()(features) if len(features) > 1 else features[0]

        return self.dense_layers(x)


class TwoTowerModel(tfrs.Model):
    """
    Unlike the individual towers above this class inherits from tensorflow-recommenders and contains boilerplate for
    training the model with sampled in-batch softmax
    """

    def __init__(
            self,
            user_model: TowerModel,
            item_model: TowerModel,
            vocab: Dict[str, np.ndarray],
            candidate_id: str,
            task: tfrs.tasks.Retrieval
    ):
        """

        Args:
            user_model: an instantiated TowerModel class for the user features
            item_model: an instantiated TowerModel class for the item features
            vocab: The same vocab generated from the ETL. Vocab should be a pickled dict with the name of the features
                as the keys and the distinct values that occurred in the dataset as the dict values.
            candidate_id: the name of the key where the candidates are stored in the input dict typically in this
                project's setup it is the same as the column name where the impressions are stored,
            task: a tfrs.tasks.Retrieval that has been setup with all the candidate data.
        """
        super().__init__()

        self.user_model = user_model
        self.item_model = item_model
        self.candidate_id = candidate_id
        self.task = task
        self.item_lookups = {}
        self.embeddings = {}
        self._user_shared_embedding_features = None
        self._item_shared_embedding_features = None

        if user_model.categorical_features is not None:
            self._user_shared_embedding_features = [f for f in user_model.categorical_features
                                                    if f.shared_embedding_key is not None]
        if item_model.categorical_features is not None:
            self._item_shared_embedding_features = [f for f in item_model.categorical_features
                                                    if f.shared_embedding_key is not None]

        if self._user_shared_embedding_features is not None:
            self._set_embeddings(self._user_shared_embedding_features, vocab)

        if self._item_shared_embedding_features is not None:
            self._set_embeddings(self._item_shared_embedding_features, vocab)

    def _set_embeddings(self, features: Dict[str, tf.Tensor], vocab: Dict[str, Dict[str, np.ndarray]]):
        """
        Used for global embeddings. not currently used in training script
        Args:
            features: a dict where the keys are either self.numerical_key or the feature names of the categorical
                features and any other information you may want to pass.
            vocab: same as vocab in __init__
        Returns:

        """
        for feature in features:
            item_lookup = tf.keras.layers.StringLookup(
                max_tokens=feature.max_tokens,
                mask_token="",
                vocabulary=list(
                    filter(lambda s: s != "", vocab[feature.shared_embedding_key].tolist())))

            self.item_lookups[feature.name] = item_lookup

            self.embeddings[feature.shared_embedding_key] = tf.keras.layers.Embedding(
                input_dim=len(vocab[feature.shared_embedding_key]) + 2,
                output_dim=feature.output_dim,
                input_length=feature.output_sequence_length,
                mask_zero=True)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Does as the method says, this is used by tensorflow recommenders. refer to tfrs.tasks.Retrieval().call()
            for more info
        Args:
            features: a dict where the keys are either self.numerical_key or the feature names of the
                categorical features.
            training: bool on whether the model is training mode.
        Returns: computed losses as a tensor
        """
        user_embeddings, item_embeddings = self(features)
        return self.task(user_embeddings, item_embeddings, compute_metrics=not training)

    def call(self, features: Dict[Text, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Like above overrides the parent method call which allows for self()
        Args:
            features: a dict where the keys are either self.numerical_key or the feature names of the
                categorical features.
        Returns: The computed representations from the user and item tower as tensors
        """
        user_embeddings = self.user_model_call(features)
        item_embeddings = self.item_model_call(features)

        return user_embeddings, item_embeddings

    def user_model_call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        embeddings = {}
        if self._user_shared_embedding_features is not None:
            x = {k.name: features[k.name] for k in self._user_shared_embedding_features}
            embeddings = self._get_embeddings(x, self._user_shared_embedding_features)

        return self.user_model(features, embeddings)

    def item_model_call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
        embeddings = {}
        if self._item_shared_embedding_features is not None:
            x = {k.name: features[k.name] for k in self._item_shared_embedding_features}
            embeddings = self._get_embeddings(x, self._item_shared_embedding_features)

        return self.item_model(features, embeddings)

    def _get_embeddings(self, x: Dict[str, tf.Tensor], features: List[CategoricalFeature]) -> Dict[str, tf.Tensor]:
        embeddings = {}

        if len(features):
            for feature in features:
                text = self.item_lookups[feature.name](x[feature.name])
                embeddings[feature.name] = self.embeddings[feature.shared_embedding_key](text)

        return embeddings

    def train_step(self, features: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
        """
        Overrides the parent train_step method so that things like passing the probabilities to the task object
            and other data is passed such as a tensor to check for duplicates under the first if statement.
        Args:
            features: a dict where the keys are either self.numerical_key or the feature names of the
                categorical features.
        Returns: computed metrics
        """
        candidate_ids, candidate_sampling_probability = None, None

        if self.candidate_id is not None and self.task._remove_accidental_hits:
            # yes we are accessing "hidden" instance variable...
            candidate_ids = features[self.candidate_id]

        if "candidate_sampling_probability" in features:
            candidate_sampling_probability = features["candidate_sampling_probability"]

        with tf.GradientTape() as tape:
            query_embeddings, item_embeddings = self(features)

            loss = self.task(query_embeddings, item_embeddings, compute_metrics=False,
                             candidate_ids=candidate_ids,
                             candidate_sampling_probability=candidate_sampling_probability)

            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
        """
        Overrides the parent test_step method so that task can be called.
        Args:
            features: a dict where the keys are either self.numerical_key or the feature names of the
                categorical features.
        Returns: computed test metrics
        """
        candidate_ids = None

        if self.candidate_id is not None and self.task._remove_accidental_hits:
            # yes we are accessing "hidden" instance variable...
            candidate_ids = features[self.candidate_id]

        query_embeddings, item_embeddings = self(features)
        loss = self.task(query_embeddings, item_embeddings, candidate_ids=candidate_ids)

        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def set_embedding_matrix(self, name: str, embedding_matrix: np.ndarray):
        self.embeddings[name].set_weights([embedding_matrix])
