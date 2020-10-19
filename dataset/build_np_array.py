import argparse
import glob
import os
import pathlib
from datetime import datetime

from helper.verificationrank import verificationRank
import numpy as np
import pandas as pd
import tensorflow as tf

import helper.crps as crps
from helper.country import convert_country
from helper.date import convert_date

parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--dataset_dir", dest="dataset_dir", metavar="FILE",
                    help="folder with glob where the dataset is")

parser.add_argument("--np_dir", dest="np_dir", metavar="FILE",
                    help="folder where to store the numpy arrays")


"""
parser.add_argument("--normalization", dest="normalization", default=False,  action='store_true',
                    help="Activate normalization of the data")
""" 

args = parser.parse_args()


def main(args):
    fileglob = glob.glob(os.path.join(args.dataset_dir, 'ecmwf_*_240.csv'))
    start = datetime.now()
    train_set = []
    train_set_all = []
    valid_set = []
    test_set = []
    print('[INFO] starting reading...')
    # read all data and concat the train set for mean and std
    for path in fileglob:
        df = pd.read_csv(path,  index_col=0)
        train = df[(df['init date'] > '1997-01-01') 
                            & (df['init date'] < '2009-12-31')]
        valid_set.append(df[(df['init date'] > '2010-01-01')
                            & (df['init date'] < '2013-12-31')].to_numpy())
        test_set.append(df[(df['init date'] > '2014-01-01')
                           & (df['init date'] < '2017-12-31')].to_numpy())

        train_set_all.append(train)
        train_set.append(train.to_numpy())
    
    print('[INFO] starting calculating mean and std...')

    # calculate mean and std
    train_set_all = pd.concat(train_set_all, axis=0).to_numpy()

    train_set_numpy = []
    for item in train_set_all:
        train_set_numpy.append(item[2:])

    train_set_all = np.array(train_set_numpy, dtype='float64')
    train_mean = np.concatenate(([0, 0], train_set_all.mean(axis=0)), axis=0)
    train_std = np.concatenate(([0, 0], train_set_all.std(axis=0)),  axis=0)

    # save the mean and std
    np.save(os.path.join(args.np_dir, 'train_mean.npy'), train_mean)
    np.save(os.path.join(args.np_dir, 'train_std.npy'), train_std)

    print('[INFO] starting normalizing and transforming to model tensor...')

    train_set = convert_to_model_data(
        train_set, train_mean, train_std)
    valid_set = convert_to_model_data(
        valid_set, train_mean, train_std)
    test_set = convert_to_model_data(
        test_set, train_mean, train_std)

    print('[INFO] starting saving dataset split..')
    np.save(os.path.join(args.np_dir, 'train_set.npy'), train_set)
    np.save(os.path.join(args.np_dir, 'valid_set.npy'), valid_set)
    np.save(os.path.join(args.np_dir, 'test_set.npy'), test_set)
    end = datetime.now()
    print(end-start)
    print('[INFO] Finished')


def convert_to_model_data(set, mean, std):
    data = []
    for i in range(len(set[0])):
        row = []
        # first element [date_transformed, country_id, AT, ZO, ZOEA, AR, ZOWE, BL, GL]
        date = convert_date(set[0][i][0])
        country = convert_country(set[0][i][1])
        matrix_data = []
        regime_data = []
        
        for file in set:
            matrix_data.append((file[i][6:25]-mean[6:25])/std[6:25])
            regime_data.append(file[i][25:32]) #(set[0][i][25:32]-mean[25:32])/std[25:32] 

        label = np.array(set[0][i][2:6], dtype='float64')

        regime_data = np.array(regime_data, dtype='float64')
        vector = np.concatenate((np.array([country, date]), regime_data.mean(axis=0), regime_data.std(axis=0)), axis=0)
        matrix_data = np.array(matrix_data, dtype='float64')

        matrix = np.array(
            [matrix_data.mean(axis=0), matrix_data.std(axis=0)], dtype='float64')
        
        ensemble = []
        for file in set:
            ensemble.append(file[i][22])

        Crps = crps.ensemble(set[0][i][2], ensemble)

        row.append(vector)
        row.append(matrix)
        row.append(label)
        row.append(Crps)
        row.append(verificationRank(set[0][i][2], ensemble))
        row.append(datetime.strptime(set[0][i][0],'%Y-%m-%d').month)
        data.append(row)
    
    print('[INFO] Finished processing data')
    return data


if __name__ == "__main__":
    if not os.path.exists(args.np_dir):
        os.makedirs(args.np_dir)
    main(args)
