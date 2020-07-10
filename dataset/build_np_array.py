import argparse
import glob
import os
import pathlib
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from helper.country import convert_country
from helper.date import convert_date
from read_csv_file import filter_country, read_csv

parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--dataset_dir", help="folder with glob where the dataset is",)

parser.add_argument("--np_dir", help="folder where to store the numpy arrays")

args = parser.parse_args()


def main(args):
    fileglob = glob.glob(args.dataset_dir)
    start = datetime.now()
    train_set = []
    train_set_all = []
    valid_set = []
    test_set  = []
    print('[info] starting reading...')
    # read all data and concat the train set for mean and std
    for path in fileglob:
        df = pd.read_csv(path,  index_col=0)
        train_set_all.append(df[(df['init date'] > '1997-01-01') & (df['init date'] < '2008-12-31')])
        train_set.append(df[(df['init date'] > '1997-01-01') & (df['init date'] < '2008-12-31')].to_numpy())
        valid_set.append(df[(df['init date'] > '2009-01-01') & (df['init date'] < '2012-12-31')].to_numpy())
        test_set.append(df[(df['init date'] > '2013-01-01') & (df['init date'] < '2017-12-31')].to_numpy())
    
    #print('[info] starting calculating mean and std...')
    ## calculate mean and std
    #train_set_all = pd.concat(train_set_all, axis=0)
    #train_mean = train_set_all.mean()
    #train_std = train_set_all.std()
    #
    #train_mean = train_mean.to_numpy()
    #train_std = train_std.to_numpy()
#
    ##save the mean and std
    #np.save(os.path.join(args.np_dir, 'train_mean.npy'), train_mean)
    #np.save(os.path.join(args.np_dir, 'train_std.npy'), train_std)
    
    train_mean = np.load(os.path.join(args.np_dir, 'train_mean.npy'))
    train_std  = np.load(os.path.join(args.np_dir, 'train_std.npy'))

    print('[info] starting normalizing and transforming to model tensor...')
    train_set = convert_to_model_data(train_set, train_mean, train_std)
    valid_set = convert_to_model_data(valid_set, train_mean, train_std)
    test_set  = convert_to_model_data(test_set, train_mean, train_std)

    print('[info] starting saving dataset split..')
    np.save(os.path.join(args.np_dir, 'train_set.npy'), train_set)
    np.save(os.path.join(args.np_dir, 'valid_set.npy'), valid_set)
    np.save(os.path.join(args.np_dir, 'test_set.npy'), valid_set)
    end = datetime.now()
    print(end-start)
    print('Finished')


def convert_to_model_data(set, mean, std):
    data = []
    for i in range(len(set[0])):
        row = []
        #first element [date_transformed, country_id, AT, ZO, ZOEA, AR, ZOWE, BL, GL]
        date = convert_date(set[0][i][0])
        country = convert_country(set[0][i][1])
        vector_data = (set[0][i][25:32]-mean[23:30])/std[23:30]
        vector = np.append([date, country ], vector_data)
        #second element are the esamlple
        matrix = []
        for file in set:
                matrix.append((file[i][6:25]-mean[4:23])/std[4:23])
        label=(np.array(set[0][i][2:6])-mean[0:4])/std[0:4]
        row.append(vector)
        row.append(np.array(matrix))
        row.append(label)
        data.append(row)
    print('finished processing data')
    return data



if __name__ == "__main__":
    main(args)
