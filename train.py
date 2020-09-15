import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD, Adam

import dataset.converter as converter
import dataset.helper.crps as crps
import dataset.shape as shape
import helper as helpers
import model.build_model as modelprovider
import model.loss_functions as loss


parser = argparse.ArgumentParser(description='This is the training script')

parser.add_argument("--exp_name", dest="name", required=True,
                    help="name of the experiment", type=str)

parser.add_argument("--data_numpy", dest="numpy_path",
                    help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

parser.add_argument("--log_dir", dest="logdir",
                    help="folder where tensorboard prints the logs", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Tests/')

parser.add_argument("--learning_rate", dest="learningrate", type=float,
                    help="Learning rate for the optimizer", default='0.002')

parser.add_argument("--batch_size", dest="batchsize", type=int,
                    help="batch size of the experiment", default='1')

parser.add_argument("--epochs", dest="epochs", type=int,
                    help="epoch count of the experiment", default='1')

parser.add_argument("--initial_epochs", dest="initialepochs", type=int,
                    help="initial epoch count of the experiment", default='0')

parser.add_argument("--model", dest="modeltype",
                    choices=['emp', 'single', 'multi'])


args = parser.parse_args()


def main(args):
    start = datetime.now()
    # get the data
    train_data = helpers.load_data(args.numpy_path, 'train_set.npy')
    valid_data = helpers.load_data(args.numpy_path, 'valid_set.npy')
    test_data = helpers.load_data(args.numpy_path, 'test_set.npy')
    test_data_labels = test_data[:, 2]
    test_data_labels = np.array([item[0] for item in test_data_labels])
    test_data_countries = test_data[:, 0]
    test_data_countries = np.array([item[0] for item in test_data_countries])
    
    test_data_feature_importance = np.random.shuffle(test_data[:,0])

    # convert the data
    train_dataset, train_shape = converter.convert_numpy_to_multi_input_dataset(
        train_data, batchsize=args.batchsize, shuffle=1000, shape=True)
    valid_dataset = converter.convert_numpy_to_multi_input_dataset(
        valid_data, batchsize=1000, shuffle=100)
    test_dataset = converter.convert_numpy_to_multi_input_dataset(
        test_data, batchsize=1000)

    model = modelprovider.build_multi_input_model(
        train_shape[1], train_shape[2])

    # Loading the model
    # Print Model
    modelprovider.printModel(model, dir=os.path.join(
        args.logdir, args.name), name=args.name+".png")

    # compiling the model
    lossfn = loss.crps_cost_function
    opt = Adam(lr=args.learningrate, amsgrad=True)
    model.compile(loss=lossfn, optimizer=opt)

    # Load model if exits
    checkpoint_dir = os.path.join(args.logdir, args.name, 'checkpoints/')

    # setup Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.logdir, args.name), update_freq='batch', histogram_freq=0, write_graph=True, write_images=False,
                                                          profile_batch=2)


    # begin with training
    print('[INFO] Starting training')
    predictions = []
    for i in range(1, 11):
        print('Round number: '+str(i))
        model = modelprovider.build_multi_input_model(
            train_shape[1], train_shape[2])
        
        model.compile(loss=lossfn, optimizer=opt)

        cp_callback_versuch = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'round-'+str(i)+'/')+"checkpoint_{epoch}", monitor='val_loss', save_weights_only=True, mode='min', verbose=0)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'round-'+str(i)+'/checkpoint'), monitor='val_loss', save_weights_only=True, mode='min', save_best_only=True, verbose=0)

        model.fit(
            train_dataset,
            epochs=args.epochs,
            initial_epoch=args.initialepochs,
            batch_size=args.batchsize,
            verbose=1,
            validation_data=valid_dataset,
            validation_batch_size=1000,
            callbacks=[tensorboard_callback, cp_callback, cp_callback_versuch],
        )
        model.load_weights(os.path.join(checkpoint_dir, 'round-'+str(i)+'/checkpoint'))
        
        predictions.append(model.predict(
            test_dataset, batch_size=1000, verbose=0))

    predictions = np.array(predictions)
    # Make sure std is positive
    predictions[:, :, 1] = np.abs(predictions[:, :, 1])
    mean_predictions = np.mean(predictions, 0)
    test_crps = crps.norm_data(test_data_labels, mean_predictions)
    #print_country(mean_predictions, test_data_countries)
    ger_data = []
    swe_data = []
    spa_data = []
    uk_data  = []
    rou_data = []
    for i in range(len(test_data_countries)):
        if test_data_countries[i]==8:
            ger_data.append(test_crps[i])
        if test_data_countries[i]==16:
            swe_data.append(test_crps[i])
        if test_data_countries[i]==2:
            spa_data.append(test_crps[i])
        if test_data_countries[i]==5:
            uk_data.append(test_crps[i])
        if test_data_countries[i]==21:
            rou_data.append(test_crps[i])

    ger_score =  round(np.array(ger_data).mean() , 2 )
    swe_score =  round(np.array(swe_data).mean() , 2 )
    spa_score =  round(np.array(spa_data).mean() , 2 )
    uk_score  =  round(np.array(uk_data).mean()  , 2 )
    rou_score =  round(np.array(rou_data).mean() , 2 )
    test_score = round(test_crps.mean()          , 2 )

    
    print(f'All test score: {test_score}')
    print(f'Ger test score: {ger_score}')
    print(f'SWE test score: {swe_score}')
    print(f'SPA test score: {spa_score}')
    print(f' UK test score: {uk_score}')
    print(f'ROU test score: {rou_score}')




    print(datetime.now()-start)

if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
