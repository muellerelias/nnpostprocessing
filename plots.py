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
import math


"""
 - all
"""

expname = 'model-1'
numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/15days/vorverarbeitetRegime/'
logdir = '/home/elias/Nextcloud/1.Masterarbeit/Tests/15days/'
batchsize = 64
epochs = 30
initial_epochs = 0
learning_rate = 0.001 #7.35274727758453e-06
train_model = False

def main():
    start = datetime.now()
    # get the data
    train_data = helpers.load_data(numpy_path, 'train_set.npy')
    valid_data = helpers.load_data(numpy_path, 'valid_set.npy')
    test_data = helpers.load_data(numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    test_data_month = test_data[:, 5]
    
    # convert the data
    train_dataset, train_shape = convert_dataset(
        train_data, batchsize=batchsize, shuffle=1000, shape=True)
    valid_dataset = convert_dataset(
        valid_data, batchsize=1000, shuffle=100)
    test_dataset = convert_dataset(
        test_data, batchsize=1000)

    print(train_shape)
    model = build_model(
        train_shape[1], train_shape[2])

    # Loading the model
    # Print Model
    modelprovider.printModel(model, dir=os.path.join(
        logdir, expname), name=expname+".png")

    # compiling the model
    lossfn = loss.crps_cost_function
    opt = Adam(lr=learning_rate, amsgrad=True)
    model.compile(loss=lossfn, optimizer=opt)

    checkpoint_dir = os.path.join(logdir, expname, 'checkpoints/')

    # begin with training
    print('[INFO] Starting training')
    predictions    = []
    predictions_ev = []
    for i in range(1, 11):
        print('Round number: '+str(i))
        model = build_model(
            train_shape[1], train_shape[2])
        
        model.compile(loss=lossfn, optimizer=opt)
        model.load_weights(os.path.join(checkpoint_dir, 'round-'+str(i)+'/best_checkpoint')).expect_partial()
        
        predictions.append(model.predict(
            test_dataset, batch_size=1000, verbose=0))
        predictions_ev.append(model.evaluate(
            test_dataset, batch_size=1000, verbose=0))

    print(predictions_ev)
    predictions = np.array(predictions)
    # Make sure std is positive
    predictions[:, :, 1] = np.abs(predictions[:, :, 1])
    mean_predictions = np.mean(predictions, 0)
    test_crps = crps.norm_data(test_data_labels, mean_predictions)
    
    print(test_crps.mean())
    pit  =[]
    rank = []
    for i in range(len(test_data)):
        pred = mean_predictions[i]
        item = test_data[i] 
        pit.append(helpers.calculatePIT(
            item[2][0], pred[0], abs(pred[1])))
        rank.append(item[4])    
    
    printHist(pit)

    """
    fig, axes = plt.subplots(1, 2, figsize=(10,3), dpi=200)
    axes[0].hist(pit,  bins=12, range=(0, 1), color='#009682', label='PIT')
    axes[1].hist(rank, bins=12, range=(1, 13), color='#009682', label='RANK', rwidth=1)
    axes[1].set_xticks([i for i in range(1, 14)])
    axes[1].set_xticklabels([str(i) for i in range(1, 14)])
    axes[1].set_title('Verification Rank (all Countries)')
    axes[0].set_title('PIT (all Countries)')
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.35)
    plt.show()

    print(('all', round(np.array(test_crps).mean() , 2 ) ))
    filter_data = np.array([])
    for i in [9,10,11]:
            filter = test_data_month==i
            test_crps = np.array(test_crps)
            filter_data = np.concatenate((filter_data, test_crps[filter]))

    print(len(filter_data))
    if len(filter_data)>0:
        item = (round(np.array(filter_data).mean() , 2 )) 
    else:
        item = ( 0)
    print( item )

    #print results
    print(datetime.now()-start)
    """

def build_model(shape_vec, shape_mat):
    # first branch for the
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(23, 22, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)
    # second branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, inp2, model3])
    # add the hiddden layers
    x = Dense( 100 , activation='relu' , name="Combined_Hidden_Layer_1" )( x )
    x = Dense( 100 , activation='relu' , name="Combined_Hidden_Layer_2" )( x )
    x = Dense( 100 , activation='relu' , name="Combined_Hidden_Layer_3" )( x )
    x = Dense(   2                     , name="Output_Layer" )(x)
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



def printHist(data):
    histo = plt.hist(data, bins=20, range=(0, 1))
    return_string = ''
    return_array  = []
    for i in range(len(histo[0])):
        return_array.append((round(histo[1][i],2),histo[0][i]))
        return_string += '('+str(round(histo[1][i],2))+','+ str(histo[0][i])+') '
    print(return_string)
    return return_array

if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(logdir, expname))
    main()
