from datetime import datetime
import glob
import os
import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

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


