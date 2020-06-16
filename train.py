import numpy as np
import os
import time

import tensorflow as tf
import dataset.read_csv_files as provider

data = provider.read_csv('ecmwf_PF_03_240.csv')


data1 = provider.filterCountry(data, 'Germany')

print(data1[505])