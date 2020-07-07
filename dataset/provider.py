import os
import pathlib
import numpy as np
import tensorflow as tf

import glob
from .read_csv_file import read_csv, filter_country

from .helper.date import convert_date
from .helper.country import convert_country


def read(filepath, country=None):
      """return the list of data
      where the output is an array of [input_vec , input_mat , output_vec]
      Returns:
      Args: 
            filepath: is the where the .csv files are stored
            country: filter by country
      Returns:
            return: [[
                        shape_vec: Shape of the input vector
                        shape_mat: Shape of the input matrix
                        shape_out: Shape of the output vector
                    ]]
      """
      
      fileglob = glob.glob(filepath) 
      files = []
      print('Read data from: '+filepath+'...')
      for csvfile in fileglob:
            if (country):
                  files.append(filter_country(read_csv(csvfile), country))
            else:
                  files.append(read_csv(csvfile))
      print('finished reading data')

      print('Proccessing data...')
      data = []

      for i in range(len(files[0])):
            row = []
            #first element [date_transformed, country_id, AT, ZO, ZOEA, AR, ZOWE, BL, GL]
            date = convert_date(files[0][i][1])
            country = convert_country(files[0][i][2])
            vector = np.array([date, country, files[0][i][26],files[0][i][27],files[0][i][28],files[0][i][29],files[0][i][30], files[0][i][31],files[0][i][32]])
            #second element are the esamlple
            matrix = []
            for file in files:
                  matrix.append(file[i][7:26])
            label=np.array(files[0][i][3:7])
            row.append(vector)
            row.append(np.array(matrix))
            row.append(label)
            data.append(row)
      print('finished processing data')
      return data


def main():
      print(read('/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/ecmwf_*_240.csv')[1])

def shape(item):
      """return the shape of data inputs
      Args: 
            item: is an array of [input_vec , input_mat , output_vec]
      Returns:
            shape_vec: Shape of the input vector
            shape_mat: Shape of the input matrix
            shape_out: Shape of the output vector
      """
      return (item[0].shape, item[1].shape, item[2].shape)

if __name__ == "__main__":
    main()