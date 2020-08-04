import argparse
import json
import os
from datetime import datetime
import kerastuner as kt

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Embedding,
                                     Flatten, Input, InputLayer)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam

import dataset.converter as converter
import dataset.helper.crps as crps
import dataset.shape as shape
import helper as helpers
import model.build_model as modelprovider
import model.loss_functions as loss


def build_model(hp):
    activation='linear'
    
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, hp.Int('Embedding size', 1, 1,
                                  default=1), name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)

    inp2 = Input(shape=(8,), name="Date_and_Regimes")
    model2 = Dense(8, activation='linear',  name="Vector_Hidden_Layer_1")(inp2)
    vector_hidden = hp.Int('Hidden Date and Regime', 0, 24, default=16)
    if (vector_hidden > 0):
        model2 = Dense(vector_hidden, activation=activation,
                       name="Vector_Hidden_Layer_2")(model2)

    # third branch for the matrix input
    inp3 = Input(shape=(2,19), name="Ensemble")
    model3 = Flatten()(inp3)
    ensemble_hidden = hp.Int('Ensebmle_Hidden_Layer', 0, 76, default=38)
    if (ensemble_hidden > 0):
        model3 = Dense(19, activation=activation,
                       name="Ensemble_Hidden_Layer")(model3)

    # concatenate the two inputs
    combined = Concatenate(axis=1)([model2, model3])

    # add the hiddden layers
    combined_hidden = hp.Int('Combined_Hidden_Layer', 0, 100, default=30)
    if (combined_hidden > 0):
        x = Dense(combined_hidden, activation=activation,
                  name="Combined_Hidden_Layer")(combined)
        x = Dense(2, activation=activation, name="Output_Layer")(x)
    else:
        x = Dense(2, activation=activation, name="Output_Layer")(combined)

    # returns the Model
    model = Model([inp1, inp2, inp3], outputs=x)
    lossfn = loss.crps_cost_function
    opt = Adam(hp.Float('learning_rate', 1e-4, 1, default=0.002, sampling='log'))
    model.compile(loss=lossfn, optimizer=opt)
    return model


"""
Start with the script
"""
start = datetime.now()
# get the data
train_data = helpers.load_data(
    '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/','train_set.npy')
valid_data = helpers.load_data(
    '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/','valid_set.npy')

countryid = 8
train_data_filterd = []
valid_data_filterd = []
for item in train_data:
    if (item[0][0]==countryid):
        train_data_filterd.append(item)

for item in train_data:
    if (item[0][0]==countryid):
        valid_data_filterd.append(item)

train_data = np.array(train_data_filterd)
valid_data = np.array(valid_data_filterd)
train_dataset, train_shape = converter.convert_numpy_to_multi_input_dataset(
    train_data, batchsize=40, shuffle=1000, shape=True)
valid_dataset = converter.convert_numpy_to_multi_input_dataset(
    valid_data, batchsize=40, shuffle=100)


tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    hyperband_iterations=20)

tuner.search(train_dataset,
             validation_data=valid_dataset,
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

tuner.results_summary()

end = datetime.now()
print(end-start)

