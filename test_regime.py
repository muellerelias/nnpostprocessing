import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD, Adam

import dataset.converter as converter
import dataset.helper.crps as crps
import dataset.shape as shape
import helper as helpers
import model.build_model as modelprovider
import model.loss_functions as loss

def main():
    numpy_path='/home/elias/Nextcloud/1.Masterarbeit/Daten/10days/vorverarbeitetRegime/'
    # get the data
    test_data = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_regime = test_data[:, 0]
    test_data_regime = np.array([item[8] for item in test_data_regime])
    test_data_month = test_data[:, 5]

    print("[INFO] predict data...")

    all_score = round( test_data_regime.mean() , 2 )
    print(('all', all_score))
    for i in range(1,13):
        filter = test_data_month==i
        #filter = np.logical_and(test_data_month==i, test_data_countries==8)
        filter_data  = test_data_regime[filter]
        #print(len(filter_data))
        if len(filter_data)>0:
            item = (i, round(np.array(filter_data).mean() , 2 )) 
        else:
            item = (i, 0)
        print( item )




if __name__ == "__main__":
    main()


"""

('all', (1.96, 281.44798306178114))
(1, 2.02)
(2, 1.69)
(3, 1.53)
(4, 1.73)
(5, 1.99)
(6, 2.24)
(7, 2.3)
(8, 2.27)
(9, 2.08)
(10, 1.75)
(11, 1.78)
(12, 2.18)

"""