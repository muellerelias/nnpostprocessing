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
    numpy_path='/Daten/vorverarbeitetNorm/'
    # get the data
    test_data = helpers.load_data(numpy_path, 'test_set.npy')

    print("[INFO] predict data...")


    all_score= inference(test_data)

    print(('all', all_score))
    for i in range(1,24):
        result = inference(test_data, countryid=i)
        print((i, result))


def inference(data, countryid=None):
    if (countryid != None):
        data_filterd = []
        for item in data:
            if (item[0][0] == countryid):
                data_filterd.append(item)
        data = np.array(data_filterd)

    crps_list = []
    crps_norm = []
    for item in data:
        #input1 = np.array([item[0][0]])[np.newaxis, :]
        #input2 = item[0][2:][np.newaxis, :]
        temp = item[1][:,16]
        calc = crps.norm(
            item[2][0], temp)
        crps_norm.append(calc)
        crps_list.append(item[3])

    crps_mean = np.array(crps_list).mean(axis=0)
    crps_norm_mean = np.array(crps_norm).mean(axis=0)

    return (crps_mean,crps_norm_mean)


if __name__ == "__main__":
    main()
