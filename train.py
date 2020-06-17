import numpy as np
import os
import time
import tensorflow as tf
import dataset.read_csv_files as provider


def main():
    data = provider.read_csv('ecmwf_PF_03_240.csv')
    print(data[1])
    data1 = provider.filter_country(data, 'Germany')
    print(provider.get_labes(data1, 0))
    print(provider.get_labes_by_country(data1, 0))


if __name__ == '__main__':
    main()