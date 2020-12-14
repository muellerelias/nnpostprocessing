import argparse
import glob
import os
import pathlib
import datetime

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


parser.add_argument("--horzion", dest="forecast_horzion", required=True,
                    help="forecast horizon: 10days or 15days", type=str)

args = parser.parse_args()


def main(args):
    if args.forecast_horzion =='10days':
        fileglob = glob.glob(os.path.join(args.dataset_dir, 'ecmwf_*_240.csv'))
    elif args.forecast_horzion =='15days':
        fileglob = glob.glob(os.path.join(args.dataset_dir, 'ecmwf_*_360.csv'))
    else:
        raise Exception("Wrong forecast horizon")

    start = datetime.datetime.now()
    train_set = []
    train_set_all = []
    valid_set = []
    test_set = []
    print('[INFO] starting reading...')
    # read all data and concat the train set for mean and std
    for path in fileglob:
        df = pd.read_csv(path,  index_col=0)
        # train set
        train = df[(df['init date'] > '1997-01-01') 
                            & (df['init date'] < '2009-12-31')]
        train_set_all.append(train) # for mean and std
        train_set.append(train.to_numpy())
        # validation set
        valid_set.append(df[(df['init date'] > '2010-01-01')
                            & (df['init date'] < '2013-12-31')].to_numpy())
        #test set
        test_set.append(df[(df['init date'] > '2014-01-01')
                           & (df['init date'] < '2017-12-31')].to_numpy())

    print('[INFO] starting calculating mean and std...')

    # calculate mean and std
    train_set_all = pd.concat(train_set_all, axis=0).to_numpy()

    # delete country and date for mean and std calculation
    train_set_numpy = []
    for item in train_set_all:
        train_set_numpy.append(item[2:])

    train_set_all = np.array(train_set_numpy, dtype='float64')
    # add to first elements because of same index like the other data
    train_mean = np.concatenate(([0, 0], train_set_all.mean(axis=0)), axis=0)
    train_std  = np.concatenate(([0, 0], train_set_all.std(axis=0)) , axis=0)

    # save the mean and std
    np.save(os.path.join(args.np_dir, 'train_mean.npy'), train_mean)
    np.save(os.path.join(args.np_dir, 'train_std.npy'), train_std)

    print('[INFO] starting transforming to model tensor...')
    # calculate each set
    train_set = convert_to_model_data(
        train_set, train_mean, train_std)
    valid_set = convert_to_model_data(
        valid_set, train_mean, train_std)
    test_set = convert_to_model_data(
        test_set, train_mean, train_std)

    # saving the sets
    np.save(os.path.join(args.np_dir, 'train_set.npy'), train_set)
    np.save(os.path.join(args.np_dir, 'valid_set.npy'), valid_set)
    np.save(os.path.join(args.np_dir, 'test_set.npy'), test_set)
    print(datetime.datetime.now()-start)
    print('[INFO] Finished')


def convert_to_model_data(set, mean, std):
    data = []
    # iterate through all records
    for i in range(len(set[0])):
        row = []
        # transforme the date
        date = convert_date(set[0][i][0])
        # transform the country string in ID
        country = convert_country(set[0][i][1])
        
        matrix_data = []
        regime_data = []
        # iterate through all 11 ensemble members
        for file in set:
            # NWP with normalisation
            matrix_data.append((file[i][6:25]-mean[6:25])/std[6:25])
            # Regimes without normalisation
            regime_data.append(file[i][25:32])  

        #All labels
        label = np.array(set[0][i][2:6], dtype='float64')

        # transform regime ensemble members to mean and std
        regime_data = np.array(regime_data, dtype='float64')
        vector = np.concatenate((np.array([country, date]), regime_data.mean(axis=0), regime_data.std(axis=0)), axis=0)

        # transform NWPs ensemble members to mean and std
        matrix_data = np.array(matrix_data, dtype='float64')
        matrix = np.array(
            [matrix_data.mean(axis=0), matrix_data.std(axis=0)], dtype='float64')

        # 2m temperatur ensample sample        
        ensemble = []
        for file in set:
            ensemble.append(file[i][22])

        # calculate raw ensemble cprs
        Crps = crps.ensemble(set[0][i][2], ensemble)

        # collecting one record
        row.append(vector)
        row.append(matrix)
        row.append(label)
        row.append(Crps)
        row.append(verificationRank(set[0][i][2], ensemble))
        row.append((datetime.datetime.strptime(set[0][i][0],'%Y-%m-%d')+datetime.timedelta(days=15)).month)
        data.append(row)
    
    print('[INFO] Finished processing data')
    return data


if __name__ == "__main__":
    if not os.path.exists(args.np_dir):
        os.makedirs(args.np_dir)
    main(args)
