import argparse
import json
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD, Adam

import dataset.helper.crps as crps
import dataset.shape as shape
import helper as helpers
import model.build_model as modelprovider
import model.loss_functions as loss
import dataset.converter as converter

parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--data_numpy", dest="numpy_path",
                    help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

parser.add_argument("--log_dir", dest="logdir",
                    help="folder where tensorboard prints the logs", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Tests/')

parser.add_argument("--name", dest="name",
                    help="name of the experiment", default='test-default')

parser.add_argument("--batch_size", dest="batchsize", type=int,
                    help="batch size of the experiment", default='1')

parser.add_argument("--epochs", dest="epochs", type=int,
                    help="epoch count of the experiment", default='1')

args = parser.parse_args()


def main(args):
    start = datetime.now()

    # get the data
    train_data = np.load(os.path.join(
        args.numpy_path, 'train_set.npy'), allow_pickle=True)
    valid_data = np.load(os.path.join(
        args.numpy_path, 'valid_set.npy'), allow_pickle=True)
    test_data = np.load(os.path.join(
        args.numpy_path, 'test_set.npy'), allow_pickle=True)

    # convert the data
    train_dataset, train_shape = converter.convert_numpy_to_one_input_dataset(
        train_data, batchsize=args.batchsize, shuffle=100, shape=True)
    valid_dataset = converter.convert_numpy_to_one_input_dataset(
        valid_data)
    test_dataset  = converter.convert_numpy_to_one_input_dataset(
        test_data)
    
    print(train_shape)

    # Loading the model
    model = modelprovider.build_one_input_model(
        train_shape)
    modelprovider.printModel(model)

    # compiling the model
    lossfn = loss.crps_cost_function
    lr = 0.007
    opt = Adam(lr=lr, decay=1e-3 / 200)
    model.compile(loss=lossfn, optimizer=opt)
    #model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # Load model if exits
    checkpoint_dir = os.path.join(args.logdir, args.name, 'checkpoints/')
    if (tf.train.latest_checkpoint(checkpoint_dir) != None):
        model.load_weights(checkpoint_dir)

    # setup Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.logdir, args.name), update_freq='batch')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_dir, save_weights_only=True, verbose=1)

    print('[INFO] Starting training')
    model.fit(
        train_dataset,
        epochs=args.epochs,
        batch_size=args.batchsize,
        verbose=0,
        callbacks=[tensorboard_callback, cp_callback],
    )


    print('[INFO] Finished training')
    end = datetime.now()
    print(end-start)



if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
