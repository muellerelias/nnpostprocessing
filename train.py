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

args = parser.parse_args()


def main(args):
    start = datetime.now()
    # get the data
    train_data = helpers.load_data(args.numpy_path, 'train_set.npy')
    valid_data = helpers.load_data(args.numpy_path, 'valid_set.npy')
    test_data = helpers.load_data(args.numpy_path, 'test_set.npy')

    # filter the data
    test_data_labels = np.array([item[0] for item in test_data[:, 2]])
    test_data_countries = np.array([item[0] for item in test_data[:, 0]])
    test_data_month = test_data[:, 5]

    # convert the data
    train_dataset, train_shape = converter.convert_numpy_to_multi_input_dataset(
        train_data, batchsize=args.batchsize, shuffle=1000, shape=True)
    valid_dataset = converter.convert_numpy_to_multi_input_dataset(
        valid_data, batchsize=1000, shuffle=100)
    test_dataset = converter.convert_numpy_to_multi_input_dataset(
        test_data, batchsize=1000)

    # build the model
    model = modelprovider.build_multi_input_model(
        train_shape[1], train_shape[2])

    # Print Model
    modelprovider.printModel(model, dir=os.path.join(
        args.logdir, args.name), name=args.name+".png")

    # compiling the model
    lossfn = loss.crps_cost_function
    opt = Adam(lr=args.learningrate, amsgrad=True)
    model.compile(loss=lossfn, optimizer=opt)

    # Load model if exits
    checkpoint_dir = os.path.join(args.logdir, args.name, 'checkpoints/')

    # begin with training 10 times
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
            callbacks=[cp_callback, cp_callback_versuch],
        )
        model.load_weights(os.path.join(checkpoint_dir, 'round-'+str(i)+'/checkpoint'))
        
        predictions.append(model.predict(
            test_dataset, batch_size=1000, verbose=0))

    # convert to numpy array
    predictions = np.array(predictions)
    # Make sure std is positive
    predictions[:, :, 1] = np.abs(predictions[:, :, 1])
    # calculate mean between the 10 results
    mean_predictions = np.mean(predictions, 0)

    # print the results with filters
    helpers.printIntCountries(test_data_labels, test_data_countries, mean_predictions)
    helpers.printHist(helpers.datasetPIT(mean_predictions, test_data_labels))
    helpers.printIntMonth(test_data_labels, test_data_month, mean_predictions)

    # save the results
    np.save(os.path.join(args.logdir, args.expname, 'prediction'), predictions)
    print(datetime.now()-start)

if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
