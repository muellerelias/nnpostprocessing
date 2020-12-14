# go to parent directory
import sys
sys.path.append('..')

import argparse
import json
import math
import os
from datetime import datetime

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
 - all
"""

numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/15days/vorverarbeitetRegime/'

def main():
    start = datetime.now()
    # get the data

    train_data = helpers.load_data(numpy_path, 'train_set.npy')
    valid_data = helpers.load_data(numpy_path, 'valid_set.npy')
    test_data  = helpers.load_data(numpy_path, 'test_set.npy')


    print( ( 'train' , '&'+str(len( train_data ))+'&'+str(round( np.array( train_data[:, 3] ).mean(), 2))))
    print( ( 'valid' , '&'+str(len( valid_data ))+'&'+str(round( np.array( valid_data[:, 3] ).mean(), 2))))
    print( ( 'test'  , '&'+str(len(  test_data ))+'&'+str(round( np.array(  test_data[:, 3] ).mean(), 2))))

    print(datetime.now()-start)

if __name__ == "__main__":
    main()
