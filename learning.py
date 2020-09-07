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

#import dataset.converter as converter
#import dataset.helper.crps as crps
#import dataset.shape as shape
#import helper as helpers
#import model.build_model as modelprovider
#import model.loss_functions as loss


def build_model(hp):
    activation='linear'
    
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, hp.Int('Embedding size', 2, 24,
                                  default=12), name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)
    emb_hidden_1 = hp.Int('Hidden_Embedding', 0, 50, default=16)
    if (emb_hidden_1 > 0):
        model1 = Dense(emb_hidden_1, activation=activation,
                        name="Emb_Hidden_Layer")(model1)

    inp2 = Input(shape=(8,), name="Date_and_Regimes")
    vector_hidden = hp.Int('Hidden_Date_and_Regime_1', 1, 150, default=16)
    model2 = Dense(vector_hidden, activation=activation,
                    name="Vector_Hidden_Layer_1")(inp2)

    # third branch for the matrix input
    inp3 = Input(shape=(2,19), name="Ensemble")
    model3 = Flatten()(inp3)
    eshidden = hp.Int('Number_Esemble_Hidden_Layer', 0, 5, default=30)
    for i in range(eshidden):
        nodes = hp.Int('Ensemble_Hidden_Layer_'+str(i), 1, 150, default=30)
        model3 = Dense(nodes, activation=activation,
                  name="Ensemble_Hidden_Layer_"+str(i))(model3)

    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, model2, model3])

    # add the hiddden layers
    chidden = hp.Int('Number_Combined_Hidden_Layer', 0, 150, default=30)
    for i in range(chidden):
        nodes = hp.Int('Combined_Hidden_Layer_'+str(i), 1, 150, default=30)
        x = Dense(nodes, activation=activation,
                  name="Combined_Hidden_Layer_"+str(i))(x)


    x = Dense(2, activation=activation, name="Output_Layer")(x)
    
    # returns the Model
    model  = Model([inp1, inp2, inp3], outputs=x)
    lossfn = "mean_squred_error"

    opt = Adam(hp.Float('learning_rate', 1e-6, 0.5, default=0.002, sampling='log'), amsgrad=True)
    model.compile(loss=lossfn, optimizer=opt)
    return model

"""
Own Tuner with batch size
"""

class MyTuner(kt.Hyperband):
    def run_trial(self, trial, *args, **kwargs):
        # You can add additional HyperParameters for preprocessing and custom training loops
        # via overriding `run_trial`
        kwargs['batch_size'] = trial.hyperparameters.Choice('batch_size', [1,100,500,750,1000])
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 50)
        list(args)
        args1 = args[0].batch(kwargs['batch_size'])
        tuple(args1)
        super(MyTuner, self).run_trial(trial, args1, **kwargs)
"""
Start with the script
"""
start = datetime.now()
# get the data
#train_data = helpers.load_data(
#    '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/','train_set.npy')
#valid_data = helpers.load_data(
#    '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/','valid_set.npy')

train_data = []
valid_data = []

#train_dataset, train_shape = converter.convert_numpy_to_multi_input_dataset(
#    train_data, shuffle=1000, shape=True)
#valid_dataset = converter.convert_numpy_to_multi_input_dataset(
#    valid_data, batchsize=1, shuffle=100)


tuner = MyTuner(
    build_model,
    #max_trials=200,
    objective='val_loss',
    max_epochs=30,
    hyperband_iterations=5,
    project_name='ganzesNetz26082020mega')

#tuner.search(train_dataset,
#             validation_data=valid_dataset,
#             epochs=5,
#             callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])

tuner.results_summary(num_trials=1)

end = datetime.now()
print(end-start)