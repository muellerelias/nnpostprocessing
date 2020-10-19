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
    #numpy_path='/home/elias/Nextcloud/1.Masterarbeit'
    numpy_path='/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNormNeu/'
    # get the data
    test_data = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    test_data_month = test_data[:, 5]

    print("[INFO] predict data...")

    all_score= inference(test_data)
    print(('all', all_score.mean()))
    for i in range(1,13):
        filter = test_data_month==i
        filter_data  = all_score[filter]
        if len(filter_data)>0:
            item = (i, round(np.array(filter_data).mean() , 2 )) 
        else:
            item = (i, 0, 0)
        print( item )


def inference(data, countryid=None):
    if (countryid != None):
        data_filterd = []
        for item in data:
            if (item[0][0] == countryid):
                data_filterd.append(item)
        data = np.array(data_filterd)

    bias = []
    for item in data:
        temp = np.array(item[1][:,16]).mean()
        temp = (temp *8.27201783941896)+282.273716956735
        bias.append(item[2][0]-temp)
    return np.array(bias)


if __name__ == "__main__":
    main()



( 1,	2.28)	1.79
( 2,	1.95)	2,25
( 3,	1.91)	1.75
( 4,	1.54)	2.23
( 5,	1.5 )	1.73
( 6,    1.43)	1.65
( 7,	1.26)	1.51
( 8,	1.25)	1.36
( 9,	1.45)	1.67
(10,	1.73)	1.75
(11,	1.85)	1.86
(12,	2.26)	1.76
	1.68	2.11
