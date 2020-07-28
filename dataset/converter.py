import numpy as np
import tensorflow as tf


def convert_numpy_to_one_input_dataset(data, batchsize=None,  shuffle=None, shape=False):
    input = []
    label = []
    for item in data:
        input1 = np.concatenate((item[0], item[1][0], item[1][1]), axis=0)
        input.append( input1.reshape(47) )
        label.append( item[2][0] )

    dataset = tf.data.Dataset.from_tensor_slices((input, label))
    
    if (shuffle != None):
        dataset = dataset.shuffle(shuffle)

    if (batchsize != None):
        dataset = dataset.batch(batchsize)

    if (shape):
        return dataset, input1.shape
    else:
        return dataset

def convert_numpy_to_multi_input_dataset(data, batchsize=None,  shuffle=None, shape=False):
    input1 = []
    input2 = []
    input3 = []
    label = []
    for item in data:
        input1.append( item[0][0] )
        input2.append(item[0][1:])
        input3.append(item[1])
        label.append(item[2][0])

    dataset_input = tf.data.Dataset.from_tensor_slices((input1, input2, input3))
    dataset_label = tf.data.Dataset.from_tensor_slices(label)

    dataset = tf.data.Dataset.zip((dataset_input, dataset_label))
    
    if (shuffle != None):
        dataset = dataset.shuffle(shuffle)

    if (batchsize != None):
        dataset = dataset.batch(batchsize)

    if (shape):
        return dataset, (input1[0].shape , input2[0].shape, input3[0].shape)
    else:
        return dataset




def convert_numpy_to_emp_input_dataset(data, batchsize=None,  shuffle=None, shape=False):
    input1 = []
    input2 = []
    label = []
    for item in data:
        input1.append( item[0][0] )
        input2.append( np.concatenate((item[0][1:], item[1][0], item[1][1]), axis=0) )
        label.append(item[2][0])

    dataset_input = tf.data.Dataset.from_tensor_slices((input1, input2))
    dataset_label = tf.data.Dataset.from_tensor_slices(label)

    dataset = tf.data.Dataset.zip((dataset_input, dataset_label))
    
    if (shuffle != None):
        dataset = dataset.shuffle(shuffle)

    if (batchsize != None):
        dataset = dataset.batch(batchsize)
    
    if (shape):
        return dataset, (input1[0].shape , input2[0].shape)
    else:
        return dataset

def denormalize(pred, mean,  std):
    return (pred*std[2])+mean[2]

def denormalizeStd(pred, mean,  std):
    return pred*std[2]
    
