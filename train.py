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
    
    # convert the data
    train_dataset, train_shape = converter.convert_numpy_to_multi_input_dataset(
        train_data, batchsize=args.batchsize, shuffle=1000, shape=True)
    valid_dataset = converter.convert_numpy_to_multi_input_dataset(
        valid_data, batchsize=1000, shuffle=100)

    model = modelprovider.build_multi_input_model(
        train_shape[1], train_shape[2])

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
    predictions = []
    for i in range(1,11):
        model.fit(
            train_dataset,
            epochs=args.epochs,
            initial_epoch=args.initialepochs,
            batch_size=args.batchsize,
            verbose=1,
            validation_data=valid_dataset,
            validation_batch_size=1000,
            callbacks=[tensorboard_callback, cp_callback, cp_callback_name],
        )
        print('[INFO] Finished training'+str(i))
        print(datetime.now()-start)
        model.load_weights(os.path.join(checkpoint_dir,'checkpoint'))
        predictions.append(model.predict())

    predictions = np.array(predictions)
    preds[:, :, 1] = np.abs(preds[:, :, 1])   # Make sure std is positive
    mean_preds = np.mean(preds, 0)
    ens_score = crps_normal(mean_preds[:, 0], mean_preds[:, 1], test_set.targets).mean()    print(f'Ensemble test score = {ens_score}')
    print(f'Ensemble test score = {ens_score}')

    

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
        input1 = np.array([item[0][0]])[np.newaxis, :]
        input2 = item[0][1:][np.newaxis, :]
        #input2 = np.array([item[0][1]])[np.newaxis, :]
        #input2 = np.array([item[2][0]])[np.newaxis, :]
        input3 = item[1][np.newaxis, :]
        prediction = model.predict([input1, input2, input3])
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
