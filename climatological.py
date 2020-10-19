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

file = '/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/ecmwf_PF_04_240.csv'
numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitet/'


def read():
    start = datetime.datetime.now()
    print('[INFO] starting reading...')
    # read all data and concat the train set for mean and std
    df = pd.read_csv(file,  index_col=0)

    test_set = df[(df['init date'] > '2014-01-01')
                  & (df['init date'] < '2017-12-31')].to_numpy()

    test_data = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    test_data_month = test_data[:, 5]

    scores    = []
    print('[INFO] starting calculating mean and std...')
    for case in test_set:
        date = datetime.datetime.strptime(case[0],'%Y-%m-%d')
        delta = datetime.timedelta(days=15)
        ensemble = []
        for year in range(1997,2012):
            start = (datetime.datetime.strptime(str(year)+'-'+date.strftime( '%m-%d' ),'%Y-%m-%d') - delta ).strftime( '%Y-%m-%d' )
            end   = (datetime.datetime.strptime(str(year)+'-'+date.strftime( '%m-%d' ),'%Y-%m-%d') + delta ).strftime( '%Y-%m-%d' )
            train_set = df[(df['init date'] > start)
                       & (df['init date'] < end)&(df['country']==case[1])].to_numpy()
            for i in train_set[:,2]:
                ensemble.append(i)
        score = ps.crps_ensemble(case[2], ensemble)
        scores.append(score)

    print(('all', round(np.array(scores).mean() , 2 ) ))
    for i in range(1,13):
        filter = test_data_month==i
        scores = np.array(scores)
        filter_data  = scores[filter]
        if len(filter_data)>0:
            item = (i, round(np.array(filter_data).mean() , 2 )) 
        else:
            item = (i, 0, 0)
        print( item )
    
    print("countries:")
    for i in [8,16,2,5,20]:
        filter = test_data_countries==i
        scores = np.array(scores)
        filter_data  = scores[filter]
        if len(filter_data)>0:
            item = (i, round(np.array(filter_data).mean() , 2 )) 
        else:
            item = (i, 0, 0)
        print( item )


if __name__ == "__main__":
    read()


"""
('all', 1.66)
(1, 2.31)
(2, 1.98)
(3, 1.62)
(4, 1.54)
(5, 1.39)
(6, 1.42)
(7, 1.24)
(8, 1.31)
(9, 1.41)
(10, 1.70)
(11, 1.89)
(12, 2.31)
countries:
(8, 1.69)
(16, 1.85)
(2, 1.24)
(5, 1.31)
(20, 1.75)
"""