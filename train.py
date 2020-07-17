import argparse
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



tf.config.threading.set_intra_op_parallelism_threads(44)

parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--data_numpy", dest="numpy_path",
        help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

parser.add_argument("--log_dir", dest="logdir",
        help="folder where tensorboard prints the logs", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Tests/')

parser.add_argument("--name", dest="name",
        help="name of the experiment", default='test-default')

args = parser.parse_args()

def main(args):
    start = datetime.now()

    # get the data
    train_data = np.load(os.path.join(args.numpy_path, 'train_set.npy'), allow_pickle=True)
    valid_data = np.load(os.path.join(args.numpy_path, 'valid_set.npy'), allow_pickle=True)
    test_data = np.load(os.path.join(args.numpy_path, 'test_set.npy'), allow_pickle=True)
    
    # get the shape
    shape_vec, shape_mat, shape_out = shape.shape(train_data[1])

    # Loading the model
    model = modelprovider.build_multi_input_model(shape_vec, shape_mat, shape_out)
    modelprovider.printModel(model)

    # compiling the model
    lossfn = loss.crps_cost_function 
    lr = 0.007
    opt = Adam(lr=lr)
    model.compile(loss = lossfn, optimizer = opt)

    #setup Tensorboard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.logdir, args.name))

    # Valid set for training
    v_input = []
    v_label = []
    np.random.shuffle(valid_data)
    for item in valid_data[:50]:
        v_input.append([item[0][np.newaxis, :],item[1][np.newaxis, :]])
        v_label.append(item[2][np.newaxis, :])
    v_set = (v_input , v_label)

    # Training the model
    print('[INFO] Starting training')
    input = []
    label = []
    for item in train_data:
        input.append([item[0][np.newaxis, :],item[1][np.newaxis, :]])
        label.append(item[2][np.newaxis, :])
    batchsize = 5000
    steps = len(label)/batchsize
    model.fit(
            x=input, 
            y=label, 
            epochs=10, 
            batch_size=batchsize,
            steps_per_epoch = steps,   
            validation_data=v_set,
            callbacks=[tensorboard_callback],
        )#            verbose=0,

    
    print('[INFO] Finished training')
    end = datetime.now()
    print(end-start)

    # Train the model
    print("[INFO] training model...")
    
    print('Test score:', model.evaluate(x=input, y=label, batch_size=4000, verbose=0 ))

    np.random.shuffle(test_data)
    print("[INFO] predict data...")
    for item in test_data[:10]:
        input1 = item[0]
        input2 = item[1]
        print([item[2][0], item[3]])
        prediction = model.predict([input1[np.newaxis, :], input2[np.newaxis, :]])
        print([prediction[0][0], prediction[0][1], crps.norm(item[2][0], prediction[0]) ])
    end = datetime.now()
    print(end-start)
    print('Finished')

if __name__ == "__main__":
    helpers.mkdir_not_exists(os.path.join(args.logdir, args.name))
    with tf.device("/cpu:0"):
        main(args)
