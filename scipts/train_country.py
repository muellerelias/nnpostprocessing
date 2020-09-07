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

parser.add_argument("--exp_name", dest="name", required=True, help="name of the experiment", type=str)

parser.add_argument("--data_numpy", dest="numpy_path",
                    help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

parser.add_argument("--log_dir", dest="logdir",
                    help="folder where tensorboard prints the logs", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Tests/')


parser.add_argument("--batch_size", dest="batchsize", type=int,
                    help="batch size of the experiment", default='1')

parser.add_argument("--learning_rate", dest="learningrate", type=float,
                    help="Learning rate for the optimizer", default='0.002')

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
    test_data  = helpers.load_data(args.numpy_path, 'test_set.npy')
    train_data = np.concatenate( (train_data , valid_data), axis=0)
    valid_data = test_data

    countryid  = 5
    train_data_filterd = []
    valid_data_filterd = []
    for item in train_data:
        if (item[0][0] == countryid):
            train_data_filterd.append(item)

    for item in train_data:
        if (item[0][0] == countryid):
            valid_data_filterd.append(item)

    train_data = np.array(train_data_filterd)
    valid_data = np.array(valid_data_filterd)

    # convert the data
    train_dataset, train_shape = converter.convert_numpy_to_country_input_dataset(
        train_data, batchsize=args.batchsize, shuffle=1000, shape=True)
    valid_dataset = converter.convert_numpy_to_country_input_dataset(
        valid_data, batchsize=args.batchsize, shuffle=100)

    model = modelprovider.build_multi_country_model(
        train_shape[0], train_shape[1])

    # Loading the model
    # Print Model
    modelprovider.printModel(model, name=args.name+".png")

    # compiling the model
    lossfn = loss.crps_cost_function
    opt = Adam(lr=args.learningrate, amsgrad=True)
    model.compile(loss=lossfn, optimizer=opt)

    # Load model if exits
    checkpoint_dir = os.path.join(args.logdir, args.name, 'checkpoints/')

    # setup Callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(args.logdir, args.name), update_freq='batch', histogram_freq=0, write_graph=True, write_images=False,
        profile_batch=2)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
         os.path.join(checkpoint_dir,'checkpoint'), monitor='val_loss', save_weights_only=True, mode='min', save_best_only=True, verbose=0)
    cp_callback_name = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(checkpoint_dir, 'all/')+"checkpoint_{epoch}", monitor='val_loss', save_weights_only=True, mode='min', verbose=0)

    # begin with training
    print('[INFO] Starting training')
    model.fit(
        train_dataset,
        epochs=args.epochs,
        initial_epoch=17,
        batch_size=args.batchsize,
        verbose=1,
        validation_data=valid_dataset,
        validation_batch_size=args.batchsize,
        use_multiprocessing=True,
        workers= 9,
        callbacks=[tensorboard_callback, cp_callback, cp_callback_name],
    )

    model.load_weights(os.path.join(checkpoint_dir,'checkpoint'))

    print('[INFO] Finished training')
    end = datetime.now()
    print(end-start)
    #result = model.evaluate(test_dataset)
    # print(result)

    print("[INFO] predict data...")

    fig, axes = plt.subplots(3, 2, figsize=(60, 20), dpi=100)

    all_pit, all_score, all_rank = inference(model, test_data)
    axes[0][0].hist(all_pit,  bins=12, range=(0, 1),  color='g')
    axes[0][1].hist(all_rank, bins=12, range=(1, 13), color='g', rwidth=1)

    # Countries: ger: 8, spain: 2, Romänien: 21, Schweden: 16, United Kingdom: 5
    ger_pit, ger_score, ger_rank = inference(model, test_data, countryid=8)
    axes[1][0].hist(ger_pit, bins=12, range=(0, 1), histtype="step", label='Germany')
    axes[1][1].hist(ger_rank, bins=12, range=(1, 13), histtype="step", rwidth=1, label='Germany')


    swe_pit, swe_score, swe_rank = inference(model, test_data, countryid=16)
    axes[1][0].hist(swe_pit, bins=12, range=(0, 1), histtype="step", label='Sweden')
    axes[1][1].hist(swe_rank, bins=12, range=(1, 13), histtype="step", label='Sweden', rwidth=1)

    spa_pit, spa_score, spa_rank = inference(model, test_data, countryid=2)
    axes[2][0].hist(spa_pit, bins=12, range=(0, 1), label="Spain", histtype="step")
    axes[2][1].hist(spa_rank, bins=12, range=(1, 13), label="Spain", histtype="step", rwidth=1)
    
    uk_pit, uk_score,  uk_rank = inference(model, test_data, countryid=5)
    axes[2][0].hist(uk_pit, bins=12, range=(0, 1), label='United Kingdom', histtype="step")
    axes[2][1].hist(uk_rank, bins=12, range=(1, 13),label='United Kingdom', histtype="step", rwidth=1)

    rom_pit, rom_score, rom_rank = inference(model, test_data, countryid=21)
    axes[1][0].hist(rom_pit, bins=12, range=(0, 1), label='Romania',  histtype="step")
    axes[1][1].hist(rom_rank, bins=12, range=(1, 13), label='Romania', histtype="step", rwidth=1)

    end = datetime.now()
    print(end-start)
    
    print(all_score)
    print(ger_score)
    print(swe_score)
    print(spa_score)
    print(uk_score)
    print(rom_score)

    axes[0][1].set_xticks([i for i in range(1, 14)])
    axes[0][1].set_xticklabels([str(i) for i in range(1, 14)])
    axes[0][1].set_title('Verification Rank (all Countries)')
    axes[0][0].set_title('PIT (all Countries)')

    axes[1][0].legend(loc='upper right')
    axes[1][0].set_title('PIT (per country)')
    axes[1][1].legend(loc='upper right')
    axes[1][1].set_xticklabels([str(i) for i in range(1, 14)])
    axes[1][1].set_title('Verification Rank (per country)')
    
    axes[2][0].legend(loc='upper right')
    axes[2][0].set_title('PIT (per country)')
    axes[2][1].legend(loc='upper right')
    axes[2][1].set_xticklabels([str(i) for i in range(1, 14)])
    axes[2][1].set_title('Verification Rank (per country)')
    
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.25, hspace=0.25)
    plt.show()


def inference(model, data, countryid=None):
    if (countryid != None):
        data_filterd = []
        for item in data:
            if (item[0][0] == countryid):
                data_filterd.append(item)
        data = np.array(data_filterd)

    pit = []
    crps_pred = []
    ranks = []
    for item in data:
        #input1 = np.array([item[0][0]])[np.newaxis, :]
        input2 = item[0][1:][np.newaxis, :]
        input3 = item[1][np.newaxis, :]
        prediction = model.predict([input2, input3])
        pred_crps = crps.norm(
            item[2][0], [prediction[0][0], abs(prediction[0][1])])
        crps_pred.append(pred_crps)
        ranks.append(item[4])
        pit.append(helpers.calculatePIT(
            item[2][0], prediction[0][0], abs(prediction[0][1])))

    crps_pred_mean = np.array(crps_pred).mean(axis=0)
    return (pit, crps_pred_mean, ranks)


if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    main(args)
