# EG Two Tower Model for Lodging Candidate Generation

## About

This repo contains code for the blog post Candidate Generation Using a Two Tower Approach With EG Traveler Data

## Install

Expedia Group makes use of DataBricks and for this reason some (interesting) libraries are required.

This library makes use of spark-tfrecord please visit
its [website](https://github.com/linkedin/spark-tfrecord) and follow the instructions to install on your machine. 

## Data
The data has been preprocessed for you and can be found **INSERT LINK HERE**.

## How to Use This Repo

### Environment
You'll first want to setup your python environment which can be done by using the ``environment-local.yml`` 
file provided in the repo. Then install eg_two_tower with pip.  
``pip install -e <path to repo dir>``

#### ETL
The code for running the data transformations are in ``eg_two_tower.etl``. We ran the ETL on DataBricks with the 
environment [Databricks Runtime 11.3 LTS for Machine Learning](https://docs.databricks.com/en/release-notes/runtime/10.4lts-ml.html).

If you want to run the ETL on your own machine you'll need either [spark-tensorflow-connector](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-connector)
or [Spark-TFRecord](https://github.com/linkedin/spark-tfrecord) in order for spark to write to tf-records. 
Since we were using DataBricks spark-tensorflow-connector is already installed. 

### Running Training 

This script is for training and replicating the experiments in the blog post.
Most of the "important" hyperparameters have defaults set with them in the script and for any other
hyper-parameters such as the size of the embeddings for the categorical features should be set by modifying the
script directly.

To run the experiments in the blog post modify these options as follows

BN + log(q) + l2_norm: : BatchNorm, log(q) correction, and L2 norm

- prob_correction = True
- numerical_input_batch_norm = True
- output_l2 True

BN + log(q): BatchNorm and log(q) correction

- prob_correction = True
- numerical_input_batch_norm = True
- output_l2 = False

BN: BatchNorm Only

- prob_correction = False
- numerical_input_batch_norm = True
- output_l2 = False

#### Quick Notes on Training
We ran the model on DataBricks using a GPU instance, but if you'd like to run it locally depending on your setup you
may want to experiment with some of the options below.

##### Hyperparameters

- **Learning Rate:** We found that the model is very sensitive to the learning rate (at least in its current setup) but
  a learning rate of **0.001** (the default for ADAM in tf) or **0.0001** should suffice (**NOTE:** we did not experiment with any
  burn-in for
  the optimizer)
- **Batch size:** we trained the model with a size of **2048** but depending on your hardware you might
  want to change this.

##### Errors

1. If you encounter any sort of OOM errors then you may want to modify the training code (train_loop)
   so that the model objects are saved (model weights), deleted, and then cleaned up but the python garbage collector.
   Additionally, you'll want to make use of ``tf.keras.backend.clear_session()``. At the time of writing this we beleive
   the error
   comes from the .eval() method of the tf.keras Model object, refer to
   this [github issue](https://github.com/tensorflow/recommenders/issues/391)
   in the official tensorflow-recommenders repo for more info.
