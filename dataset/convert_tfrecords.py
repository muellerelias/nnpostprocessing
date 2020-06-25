import os
import pathlib
import numpy as np
import tensorflow as tf

import glob
import read_csv_files as provider

import helper.date as dateconverter
import helper.country as countryconverter

filepath = '/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/ecmwf_*_240.csv'
fileglob = glob.glob(filepath) 
mistakes = 0
files = []
print('Read data from: '+filepath+'...')
for csvfile in fileglob:
      files.append(provider.read_csv(csvfile))
print('finished reading data')

print('Proccessing data...')
data = []

for i in range(len(files[0])):
      row = []
      #first element [date_transformed, country_id, AT, ZO, ZOEA, AR, ZOWE, BL, GL]
      date = dateconverter.convert_date(files[0][i][1])
      country = countryconverter.convert_country(files[0][i][2])
      vector = [date, country, files[0][i][26],files[0][i][27],files[0][i][28],files[0][i][29],files[0][i][30], files[0][i][31],files[0][i][32]]
      #second element are the esamlple
      matrix = []
      for file in files:
            matrix.append(file[i][7:26])
      label=files[0][i][3:7]
      row.append(vector)
      row.append(matrix)
      row.append(label)
      data.append(row)

print('finished processing data')