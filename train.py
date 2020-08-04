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
            train_data, batchsize=args.batchsize, shuffle=1000, shape=True)
        valid_dataset = converter.convert_numpy_to_multi_input_dataset(
            valid_data, batchsize=args.batchsize, shuffle=100)
        test_dataset = converter.convert_numpy_to_multi_input_dataset(
            test_data, batchsize=args.batchsize)
        model = modelprovider.build_multi_input_model(
            train_shape[1], train_shape[2])
    elif (args.modeltype == 'emp'):
        train_dataset, train_shape = converter.convert_numpy_to_emp_input_dataset(
            train_data, batchsize=args.batchsize, shuffle=100, shape=True)
        valid_dataset = converter.convert_numpy_to_emp_input_dataset(
            valid_data, batchsize=args.batchsize, shuffle=100)
        test_dataset = converter.convert_numpy_to_emp_input_dataset(
            test_data, batchsize=args.batchsize)
        model = modelprovider.build_emb_model(train_shape[1])
    else:
        train_dataset, train_shape = converter.convert_numpy_to_one_input_dataset(
            train_data, batchsize=args.batchsize, shuffle=100, shape=True)
        valid_dataset = converter.convert_numpy_to_one_input_dataset(
            valid_data, batchsize=args.batchsize, shuffle=100)
        test_dataset = converter.convert_numpy_to_one_input_dataset(
            test_data, batchsize=args.batchsize)
        model = modelprovider.build_one_input_model(train_shape)

    # Loading the model
    # Print Model
    modelprovider.printModel(model, name=args.modeltype+"_model.png")

    # compiling the model
    lossfn = loss.crps_cost_function
    opt = Adam(lr=0.0004)
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
    result = model.evaluate(test_dataset)
    print(result)


    # np.random.shuffle(test_data)
    print("[INFO] predict data...")
    #crps_label = []
    #crps_pred = []
    #pit = []
    #rank = []

    #mean = helpers.load_data(args.numpy_path, 'train_mean.npy')
    #std = helpers.load_data(args.numpy_path, 'train_std.npy')

    # for item in test_data[:1000]:
    #    input1 = np.array([item[0][0]])[np.newaxis, :]
    #    input2 = item[0][1:][np.newaxis, :]
    #    input3 = item[1][np.newaxis, :]
    #    prediction = model.predict([input1, input2, input3])
    #    pred_crps = crps.norm(
    #        item[2][0], [prediction[0][0], abs(prediction[0][1])])
    #    crps_label.append(item[3])
    #    crps_pred.append(pred_crps)
    #    pit.append(helpers.calculatePIT(
    #        item[2][0], prediction[0][0], abs(prediction[0][1])))
    #    rank.append(item[4])

    #crps_label_mean = np.array(crps_label).mean(axis=0)
    #crps_pred_mean = np.array(crps_pred).mean(axis=0)
    # print(crps_label_mean)
    # print(crps_pred_mean)

    end = datetime.now()
    print(end-start)

    #fig, axes = plt.subplots(1, 2, figsize=(10, 2.5), dpi=100)

    #axes[0].hist(pit, bins=12, range=(0, 1), color='g')
    # axes[0].set_title('PIT')
    # axes[1].hist(rank, bins=12, range=(1, 13),
    #             color='g', histtype="step", rwidth=1)
    #axes[1].set_xticks([i for i in range(1, 14)])
    #axes[1].set_xticklabels([str(i) for i in range(1, 14)])
    #axes[1].set_title('Verification Rank')
    # axes[1].set_xlim([1,12])
    # axes[1].set_ylim([0,150])

    # plt.show()


if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
