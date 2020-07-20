import tensorflow as tf 
import numpy as np


input = []
label = []
for i in range(1):
    input1 = np.random.rand(9)
    input2 = np.random.rand(2,19)
    label1  = np.random.rand(4)
    input.append([input1 , input2])
    label.append(label1) 

dataset = tf.data.Dataset.from_tensor_slices((input, label))