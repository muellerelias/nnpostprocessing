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

parser.add_argument("--model", dest="modeltype",
                    choices=['emp', 'single', 'multi'])


args = parser.parse_args()


def main(args):
    start = datetime.now()
    # get the data
    train_data = helpers.load_data(args.numpy_path, 'train_set.npy')
    valid_data = helpers.load_data(args.numpy_path, 'valid_set.npy')
    test_data = helpers.load_data(args.numpy_path, 'test_set.npy')

    # convert the data
    if (args.modeltype == 'multi'):
        train_dataset, train_shape = converter.convert_numpy_to_multi_input_dataset(
            train_data, batchsize=args.batchsize, shuffle=100, shape=True)
        valid_dataset = converter.convert_numpy_to_multi_input_dataset(
            valid_data, batchsize=args.batchsize, shuffle=100)
        model = modelprovider.build_multi_input_model(
            train_shape[1], train_shape[2])
    elif (args.modeltype == 'emp'):
        train_dataset, train_shape = converter.convert_numpy_to_emp_input_dataset(
            train_data, batchsize=args.batchsize, shuffle=100, shape=True)
        valid_dataset = converter.convert_numpy_to_emp_input_dataset(
            valid_data, batchsize=args.batchsize, shuffle=100)
        model = modelprovider.build_emb_model(train_shape[1])
    else:
        train_dataset, train_shape = converter.convert_numpy_to_one_input_dataset(
            train_data, batchsize=args.batchsize, shuffle=100, shape=True)
        valid_dataset = converter.convert_numpy_to_one_input_dataset(
            valid_data, batchsize=args.batchsize, shuffle=100)
        model = modelprovider.build_one_input_model(train_shape)

    # Loading the model
    # Print Model
    modelprovider.printModel(model, name=args.modeltype+"_model.png")

    # compiling the model
    lossfn = loss.crps_cost_function
    opt = Adam(lr=0.1, decay=1 / 200)
    model.compile(loss=lossfn, optimizer=opt)

    # Load model if exits
    checkpoint_dir = os.path.join(args.logdir, args.name, 'checkpoints/')
    if (tf.train.latest_checkpoint(checkpoint_dir) != None):
        model.load_weights(checkpoint_dir)

    # setup Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.logdir, args.name), update_freq='batch', histogram_freq=0, write_graph=True, write_images=False,
        profile_batch=2)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_dir, save_weights_only=True, verbose=0)

    # begin with training
    print('[INFO] Starting training')
    model.fit(
        train_dataset,
        epochs=args.epochs,
        batch_size=args.batchsize,
        verbose=1,
        validation_data=valid_dataset,
        validation_steps=20,
        callbacks=[tensorboard_callback, cp_callback],
    )

    print('[INFO] Finished training')
    end = datetime.now()
    print(end-start)

    np.random.shuffle(test_data)
    print("[INFO] predict data...")
    mean = helpers.load_data(args.numpy_path, 'train_mean.npy')
    std = helpers.load_data(args.numpy_path, 'train_std.npy')
    for item in test_data[:100]:
        result = []
        input1 = np.array([item[0][0]])[np.newaxis, :]
        input2 = item[0][1:][np.newaxis, :]
        input3 = item[1][np.newaxis, :]
        prediction = model.predict([input1, input2, input3])
        label = converter.denormalize(item[2][0], mean,  std)
        pred = converter.denormalize(prediction[0][0], mean,  std)
        pred_std = converter.denormalizeStd(prediction[0][1], mean,  std)
        pred_crps = crps.norm(label, [pred, pred_std])
        result.append([label, item[3]])
        result.append([pred, pred_std, pred_crps ])
        print(result)

if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
