import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Flatten,
                                     Input, InputLayer, Embedding)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam

import dataset.converter as converter
import dataset.helper.crps as crps
import dataset.shape as shape
import helper as helpers
import model.build_model as modelprovider
import model.loss_functions as loss


def main():
    label = []
    input = []
    while (i<=100):
        label.append(np.random.rand(2))
        input.append([np.random.rand(8), np.random.rand(11,49) ])
        i+=1
    
    
    model = build_model(8, )

    model.fit(
        
    )

def build_model(n_features , ):
    features_in = Input(shape=(n_features,))

    id_in = Input(shape=
    emb = Embedding(12, emb_size)(id_in)
    emb = Flatten()(emb)
    x = Concatenate()([features_in, emb])
    x = Dense(n_outputs, activation='linear', kernel_regularizer=reg)(x)
    model = Model(inputs=[features_in, id_in], outputs=x)

if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
