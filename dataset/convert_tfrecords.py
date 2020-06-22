import os
import pathlib

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import pandas as pd


np.set_printoptions(precision=4)

data  = [1,2,3,4,5,5]
labels = [1,2,2,2,2,2]
dataset = tf.data.Dataset.from_tensor_slices((data, labels))

for elem in dataset:
      print(elem.numpy())

