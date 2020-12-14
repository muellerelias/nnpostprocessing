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
    print('[INFO] starting reading...')
    # read all data and concat the train set for mean and std
    df = pd.read_csv(file,  index_col=0)

    test_set = df[(df['init date'] > '2001-01-01')
                  & (df['init date'] < '2004-12-31')].to_numpy()
    test_set_country = test_set[:,1]
    
    scores    = []
    print('[INFO] starting calculating mean and std...')
    for case in test_set:
        date = datetime.datetime.strptime(case[0],'%Y-%m-%d')
        delta = datetime.timedelta(days=15)
        ensemble = []
        for year in range(2005,2017+1):
            start = (datetime.datetime.strptime(str(year)+'-'+date.strftime( '%m-%d' ),'%Y-%m-%d') - delta ).strftime( '%Y-%m-%d' )
            end   = (datetime.datetime.strptime(str(year)+'-'+date.strftime( '%m-%d' ),'%Y-%m-%d') + delta ).strftime( '%Y-%m-%d' )
            train_set = df[(df['init date'] > start)
                       & (df['init date'] < end)&(df['country']==case[1])].to_numpy()
            for i in train_set[:,2]:
                ensemble.append(i)
        score = ps.crps_ensemble(case[2], ensemble)
        scores.append(score)

    to_print = str(round(np.array(scores).mean() , 2 ))
    for i in ['Germany', 'Sweden', 'Spain', 'United Kingdom', 'Romania']:
        filter = test_set_country==i
        scores = np.array(scores)
        filter_data  = scores[filter]
        if len(filter_data)>0:
            to_print = to_print +'&'+str(round(np.array(filter_data).mean() , 2 ))
        else:
            to_print = to_print +'&0'
    
    print(to_print)


if __name__ == "__main__":
    start = datetime.datetime.now()
    read()
    print(datetime.datetime.now()-start)