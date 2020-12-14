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
    numpy_path='/home/elias/Nextcloud/1.Masterarbeit/Daten/15days/vorverarbeitetRegime/'
    # get the data
    test_data = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    test_data_month = test_data[:, 5]

    print("[INFO] predict data...")

    all_score= inference(test_data)

    print(('all', all_score ))
    for i in [8,16,2,5,20]:
        result = inference(test_data, countryid=i)
        print((i, result))

    print(('all', all_score))
    for i in range(1,13):
        filter = test_data_month==i
        filter_data  = test_data[filter]
        if len(filter_data)>0:
            item = (i, round(np.array(filter_data[:,3]).mean() , 2 )) 
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

    crps_list = []
    for item in data:
        crps_list.append(item[3])

    crps_mean = np.array(crps_list).mean(axis=0)

    return round(crps_mean,2)

if __name__ == "__main__":
    main()
