# go to parent directory
import sys
sys.path.append('..')

import argparse
import datetime
import glob
import os
import pathlib
import helper as helpers

import numpy as np
import pandas as pd
import properscoring as ps
import tensorflow as tf

import dataset.helper.crps as crps
from dataset.helper.country import convert_country
from dataset.helper.date import convert_date
from dataset.helper.verificationrank import verificationRank

file = '/home/elias/Nextcloud/1.Masterarbeit/Daten/15days/2020_MA_Elias/ecmwf_PF_04_360.csv'
numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/15days/vorverarbeitetRegime/'


def read():
    start = datetime.datetime.now()
    print('[INFO] starting reading...')
    # read all data and concat the train set for mean and std
    df = pd.read_csv(file,  index_col=0)

    test_set = df[(df['init date'] > '2014-01-01')
                  & (df['init date'] < '2017-12-31')].to_numpy()

    test_data   = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    test_data_month   = test_data[:, 5]

    scores = []
    ranks  = []
    length = []
    print('[INFO] starting calculating')
    for case in test_set:
        date = datetime.datetime.strptime(case[0],'%Y-%m-%d')
        delta = datetime.timedelta(days=16)
        ensemble = []
        for year in range(1998,2013):
            start = (datetime.datetime.strptime(str(year)+'-'+date.strftime( '%m-%d' ),'%Y-%m-%d') - delta ).strftime( '%Y-%m-%d' )
            end   = (datetime.datetime.strptime(str(year)+'-'+date.strftime( '%m-%d' ),'%Y-%m-%d') + delta ).strftime( '%Y-%m-%d' )
            train_set = df[(df['init date'] > start)
                       & (df['init date'] < end)&(df['country']==case[1])].to_numpy()
            for i in train_set[:,2]:
                ensemble.append(i)
        score = ps.crps_ensemble(case[2], ensemble)
        rank  = verificationRank(case[2], ensemble)
        length.append(len(ensemble))
        ranks.append(rank)
        scores.append(score)

    print(max(ranks))
    print(list(dict.fromkeys(length)))
    scores = np.array(scores)
    helpers.printHist(ranks, r=(0,276))
    print(('all', round(scores.mean() , 2 ) ))
    result = str(scores.mean())+'&'
    for i in [8,16,2,5,20]:
        filter = test_data_countries == i
        filter_data = scores[filter]
        if len(filter_data) > 0:
            item = str(round(np.array(filter_data).mean(), 2))
        else:
            item = str(0)
        result +=  item+'&'
    print(result)

    for i in range(1, 13):
        filter = test_data_month == i
        filter_data = scores[filter]
        if len(filter_data) > 0:
            item = (i, round(np.array(filter_data).mean(), 2))
        else:
            item = (i, 0, 0)
        print(item)    


if __name__ == "__main__":
    read()
