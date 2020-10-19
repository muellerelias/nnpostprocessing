import argparse
import datetime
import glob
import os
import pathlib
import helper as helpers
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import properscoring as ps
import tensorflow as tf

import dataset.helper.crps as crps
from dataset.helper.country import convert_country
from dataset.helper.date import convert_date
from dataset.helper.verificationrank import verificationRank

file = '/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/ecmwf_PF_03_240.csv'
numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitet/'


def read():
    start = datetime.datetime.now()
    print('[INFO] starting reading...')
    # read all data and concat the train set for mean and std
    df = pd.read_csv(file,  index_col=0).to_numpy()

    days = []
    years = []
    print('[INFO] starting calculating mean and std...')
    for case in df:
        date = datetime.datetime.strptime(case[0], '%Y-%m-%d').date()
        if date.weekday() in [0,1,2,3,4,5,6]:
            years.append(date.year)



    for i in range(1997,2018):
            print(i,years.count(i)/22)


if __name__ == "__main__":
    read()