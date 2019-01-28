"""Sample LSTM Code

This script runs a generic LSTM network on the features in your training data and
makes predictions based on your test data.  It will output a model, and save and
load your model.

    * add_layer - adds a layer to the model
    * build - builds a model when passed an array of layers
    * load - loads an existing model and weights
    * run - uses above functions to build a model on training data and saves it to file
"""

import numpy as np
import process
import tensorflow as tf

from ann_visualizer.visualize import ann_viz
from functools import reduce
from keras import backend as K, regularizers
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense, InputLayer, Dropout, Flatten
from keras.metrics import categorical_accuracy
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, Adam
from math import ceil
from time import time

def add_layer(model, nodes):
    activation = 'relu' if nodes != 1 else 'linear'
    name = 'Layer{layer_num}'.format(layer_num=len(model.layers))
    with tf.name_scope(name):
        layer = Dense(
            units=nodes,
            activation=activation,
        ) if len(model.layers) else LSTM(nodes,
                                         input_shape=(1, nodes),
                                         return_sequences=True,
                                         go_backwards=True,
                                         dropout=0.3,
                                         recurrent_dropout=0.3,
                                         )

        needs_flatten = not len(model.layers)
        model.add(layer)
        if needs_flatten:
            model.add(Flatten())

    return model


def build(layers, predict_value):
    model = Sequential()

    model = reduce(add_layer, layers, model)
    loss = 'mean_squared_error' if predict_value else 'binary_crossentropy'
    metrics = ['accuracy'] if predict_value else [categorical_accuracy]
    with tf.name_scope("loss"):
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss=loss, metrics=metrics)

    return model

def load(json_file='model_lstm.json', h5_file='model_lstm.h5'):
    json = open(json_file, 'r')
    model_json = json.read()
    json.close()

    model = model_from_json(model_json)
    model.load_weights(h5_file)

    with tf.name_scope("loss"):
        optimizer = Adam(lr=0.001)
        model.compile(optimizer=optimizer,
                      loss='mean_squared_error', metrics=['accuracy'])

    return model

def run(data=None, predict_value=False, epochs=5):
    """Trains a LSTM network on inputed data
    Parameters
    ----------
    data : DataSet
        DataSet object filled with scaled and split data ready for consumption
    predict_values : Boolean
        Set to true if you want to predict the value of the stock or false if you want to categorize it
    epochs : Int
        How many epochs you want the network to run
    Returns
    -------
    keras.engine.sequential.Sequential
        A keras model we can evaluate and make predictions on
    """
    if data is None:
        data = process.run(None, 'MSFT', predict_value)

    K.clear_session()

    num_features = data.X_train.shape[1]
    first_layer = ceil(num_features / 2)
    second_layer = ceil(first_layer / 2)

    model = build([num_features, 1], predict_value)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()),
                              histogram_freq=1,
                              batch_size=5,
                              write_graph=True,
                              write_grads=True,
                              write_images=False,
                              embeddings_freq=0,
                              embeddings_layer_names=None,
                              embeddings_metadata=None
                              )

    y_labelled = data.y_train * 1
    print(model.summary())

    # reshape according to number of timesteps (1 in this case)
    data.group_timesteps(1)

    if (predict_value):
        model.fit(
            data.X_train,
            y_labelled,
            batch_size=1,
            epochs=epochs,
            validation_split=0.3,
            shuffle=False,
            callbacks=[tensorboard],
        )
    else:
        model.fit(
            data.X_train,
            y_labelled,
            batch_size=1,
            epochs=epochs,
            validation_split=0.3,
            shuffle=False,
            class_weight=data.class_weights(),
            callbacks=[tensorboard],
        )

    # ann_viz(model, view=True, filename="network.gv", title="MyNeural Network")

    json = model.to_json()
    with open("model_lstm.json", "w") as file:
        file.write(json)
    model.save_weights("model_lstm.h5")

    return model
