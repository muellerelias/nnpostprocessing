import csv
import os
from datetime import date, datetime, timedelta
from itertools import groupby
import pandas as pd
import glob
from datetime import datetime


"""
#####
This file is used to read and convert the data from the CSV file. 
#####
"""

def read_csv(path):
    return pd.read_csv(path)

def get(filepath):
    fileglob = glob.glob(filepath) 
    alldata = []
    start = datetime.now().timestamp()
    print('Calculate mean and std...')
    for csvfile in fileglob:
        alldata.append(pd.read_csv(csvfile))
    datastream = pd.concat(alldata, axis=0, ignore_index=True)
    mean = datastream.mean()
    std = datastream.std()
    end = datetime.now().timestamp()
    print(start)
    print(end)
    time = end-start
    print('Finished with "Calculate mean and std" in' + str(time))
    return [mean.to_numpy(), std.to_numpy()]

def main():
    mean, std = get('/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/ecmwf_*_240.csv')   
    print(mean)
    print(std)

if __name__ == "__main__":
    main()
