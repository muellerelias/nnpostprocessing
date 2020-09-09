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


"""
 - all, but no embeddings
"""

def build_model(hp):
    activation='linear'
    
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 23, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)

    inp2 = Input(shape=(8,), name="Date_and_Regimes")

    # third branch for the matrix input
    inp3 = Input(shape=(2,19), name="Ensemble")
    model3 = Flatten()(inp3)

    # concatenate the two inputs
    x = Concatenate(axis=1)([inp2, model3])

    # add the hiddden layers
    nodes = 100
    numb= 3
    if(numb>0):
        for i in range(1,numb):
            x = Dense(nodes, activation=activation,
                        name="Combined_Hidden_Layer_"+str(i))(x)
    
    x = Dense(2, activation=activation, name="Output_Layer")(x)
    
    # returns the Model
    model  = Model([inp1, inp2, inp3], outputs=x)
    lossfn = loss.crps_cost_function

    opt = Adam(hp.Float('learning_rate', 1e-8, 0.5, default=0.1025524430168661, sampling='log'), amsgrad=True)
    model.compile(loss=lossfn, optimizer=opt)
    return model

"""
Own Tuner with batch size
"""

class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = 1
        list(args)
        args1 = args[0].batch(kwargs['batch_size'])
        tuple(args1)
        super(MyTuner, self).run_trial(trial, args1, **kwargs)

"""
Start with the script
"""

start = datetime.now()
# get the data
train_data = helpers.load_data(
    '/root/Daten/vorverarbeitetNorm/','train_set.npy')
valid_data = helpers.load_data(
    '/root/Daten/vorverarbeitetNorm/','valid_set.npy')

train_dataset = converter.convert_numpy_to_multi_input_dataset(
    train_data, shuffle=1000)
valid_dataset = converter.convert_numpy_to_multi_input_dataset(
    valid_data, batchsize=1000, shuffle=100)


tuner = MyTuner(
    build_model,
    objective='val_loss',
    max_epochs=30,
    hyperband_iterations=10,
    project_name='ganzesNetz09092020_2')

tuner.search(train_dataset,
             validation_data=valid_dataset,
             epochs=10,
             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

tuner.results_summary(num_trials=1)

end = datetime.now()
print(end-start)