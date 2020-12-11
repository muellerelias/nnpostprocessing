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
 - all, but no embeddings
"""

expname = 'model-2'
forecast = '10days'
numpy_path = '/home/elias/Nextcloud/1.Masterarbeit/Daten/'+forecast+'/vorverarbeitetRegime/'
logdir = '/home/elias/Nextcloud/1.Masterarbeit/Tests/'+forecast+'/'
"""
batchsize = 32
epochs = 30
initial_epochs = 10
learning_rate = 0.0001
train_model = False
"""
batchsize = 16
epochs = 30
initial_epochs = 0
learning_rate = 5e-05
train_model = True

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
    
    # convert the data
    train_dataset, train_shape = convert_dataset(
        train_data, batchsize=batchsize, shuffle=1000, shape=True)
    valid_dataset = convert_dataset(
        valid_data, batchsize=1000, shuffle=100)
    test_dataset = convert_dataset(
        test_data, batchsize=1000)

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

    # Load model if exits
    checkpoint_dir = os.path.join(logdir, expname, 'checkpoints/')

    # setup Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(logdir, expname), update_freq='batch', histogram_freq=0, write_graph=True, write_images=False,
                                                          profile_batch=2)


    # begin with training
    print('[INFO] Starting training')
    predictions = []
    for i in range(1, 11):
        print('Round number: '+str(i))
        model = build_model(
            train_shape[1], train_shape[2])
        
        model.compile(loss=lossfn, optimizer=opt)

        cp_callback_versuch = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'round-'+str(i)+'/')+"checkpoint_{epoch}", monitor='val_loss', save_weights_only=True, mode='min', verbose=0)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'round-'+str(i)+'/checkpoint'), monitor='val_loss', save_weights_only=True, mode='min', save_best_only=True, verbose=0)

        if train_model:
            model.fit(
                train_dataset,
                epochs=epochs,
                initial_epoch=initial_epochs,
                batch_size=batchsize,
                verbose=1,
                validation_data=valid_dataset,
                validation_batch_size=1000,
                callbacks=[tensorboard_callback, cp_callback, cp_callback_versuch],
            )
        model.load_weights(os.path.join(checkpoint_dir, 'round-'+str(i)+'/checkpoint')).expect_partial()
        
        predictions.append(model.predict(
            test_dataset, batch_size=1000, verbose=0))

    predictions = np.array(predictions)
    # Make sure std is positive
    predictions[:, :, 1] = np.abs(predictions[:, :, 1])
    mean_predictions = np.mean(predictions, 0)

    #print stuff
    helpers.printIntCountries(test_data_labels, test_data_countries , mean_predictions)
    helpers.printHist(helpers.datasetPIT(mean_predictions, test_data_labels))

    np.save(os.path.join(logdir, expname, 'prediction'), predictions)
    print(datetime.now()-start)

def build_model(shape_vec, shape_mat):
    # first branch for the
    #inp1 = Input(shape=(1,), name='Country_ID')
    #model1 = Embedding(23, 22, name='Country_Embedding')(inp1)
    #model1 = Flatten()(model1)
    # second branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    # concatenate the two inputs
    x = Concatenate(axis=1)([inp2, model3])
    # add the hiddden layers
    x = Dense(100, activation='linear'  , name="Combined_Hidden_Layer_1")(x)
    x = Dense(100, activation='relu'    , name="Combined_Hidden_Layer_2")(x)
    x = Dense(100, activation='relu'    , name="Combined_Hidden_Layer_3")(x)
    x = Dense(  2, activation='linear'  , name="Output_Layer")(x)
    # returns the Model
    return Model([ inp2, inp3], outputs=x)

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

    dataset_input = tf.data.Dataset.from_tensor_slices((input2, input3))
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