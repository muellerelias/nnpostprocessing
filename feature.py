import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.layers import (Activation, Concatenate, Dense, Embedding,
                                     Flatten, Input, InputLayer)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam

import dataset.converter as converter
import dataset.helper.crps as crps
import dataset.shape as shape
import helper as helpers
import model.build_model as modelprovider
import model.loss_functions as loss

"""
Feature importance Country
"""

expname = 'versuch-1'
numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/'
logdir = '/home/elias/Nextcloud/1.Masterarbeit/Tests/'
batchsize = 1
epochs = 30
initial_epochs = 0
learning_rate = 7.35274727758453e-06

def main():
    start = datetime.now()
    # get the data
    test_data = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    
    # convert the data
    test_dataset = convert_dataset(
        test_data, batchsize=1000)

    # Load model if exits
    checkpoint_dir = os.path.join(logdir, expname, 'checkpoints/')
    # begin with training
    print('[INFO] Starting training')
    predictions = []
    predictions_feature = []
    for i in range(1, 11):
        print('Round number: '+str(i))
        model = build_model(
            (8,), (2,19))
        
        model.compile(loss=loss.crps_cost_function, optimizer=Adam())

        test_dataset_feature = convert_dataset_feature_importance(
            test_data, batchsize=1000)

        model.load_weights(os.path.join(checkpoint_dir, 'round-'+str(i)+'/checkpoint')).expect_partial()
        
        predictions.append(model.predict(
            test_dataset, batch_size=1000, verbose=0))
        predictions_feature.append(model.predict(
            test_dataset_feature, batch_size=1000, verbose=0))


    predictions = np.array(predictions)
    predictions_feature = np.array(predictions_feature)

    # Make sure std is positive
    predictions[:, :, 1] = np.abs(predictions[:, :, 1])
    predictions_feature[:, :, 1] = np.abs(predictions_feature[:, :, 1])

    mean_predictions = np.mean(predictions, 0)
    mean_predictions_feature = np.mean(predictions_feature, 0)

    test_crps = crps.norm_data(test_data_labels, mean_predictions)
    test_crps_feature = crps.norm_data(test_data_labels, mean_predictions_feature)
    test_score = round((1-test_crps_feature.mean()/test_crps.mean()   )*100     , 2 )
    print(test_crps_feature.mean())
    result ='&'+str(test_score)+'\%' 
    for i in [8,16,2,5,21]:
        filter = test_data_countries==i
        filter_data = test_crps_feature[filter]
        filter_test = test_crps[filter]
        item = round((1 -np.array(filter_data).mean()/np.array(filter_test).mean() )*100, 2 )
        result = result +'&'+str(item)+'\%'

    print(result)
    print(datetime.now()-start)

def build_model(shape_vec, shape_mat):
    # first branch for the
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 23, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)
    # second branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, inp2, model3])
    # add the hiddden layers
    x = Dense( 100 , activation='linear' , name="Combined_Hidden_Layer_1" )( x )
    x = Dense( 100 , activation='linear' , name="Combined_Hidden_Layer_2" )( x )
    x = Dense( 100 , activation='linear' , name="Combined_Hidden_Layer_3" )( x )
    x = Dense(   2 , activation='linear' , name="Output_Layer" )(x)
    # returns the Model
    return Model([inp1, inp2, inp3], outputs=x)

def convert_dataset(data, batchsize=None,  shuffle=None, shape=False):
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


def convert_dataset_feature_importance(data, batchsize=None,  shuffle=None, shape=False):
    input1 = []
    input21 = []
    input22 = []
    input2 = []
    input3 = []
    label = []
    for item in data:
        input1.append( item[0][0] )
        input21.append(np.array([item[0][1]] ))
        input22.append(np.array(item[0][2:]))
        #input2.append(item[0][1:])
        input3.append(item[1])
        label.append(item[2][0])

    input21 = np.array(input21)
    input22 = np.array(input22)
    np.random.shuffle(input22[:,0])
    np.random.shuffle(input22[:,1])
    np.random.shuffle(input22[:,2])
    np.random.shuffle(input22[:,3])
    np.random.shuffle(input22[:,4])
    np.random.shuffle(input22[:,5])
    #np.random.shuffle(input22[:,6])

    input2  = np.concatenate((input21, input22), axis=1)

    input3 = np.array(input3)
    #np.random.shuffle(input3)
    np.random.shuffle(input3[:,:,16][:,1])

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



if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(logdir, expname))
    main()
"""
hyper: regime individual:1.588302287788153
regime individual: 1.5185716741396715
"""