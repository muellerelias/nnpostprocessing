import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD, Adam

import dataset.helper.crps as crps
import dataset.shape as shape
import model.build_model as modelprovider
import model.loss_functions as loss

tf.config.threading.set_intra_op_parallelism_threads(44)

parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--data_numpy", dest="numpy_path",
        help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

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

    # Training the model
    for i in range(10):
        print('[INFO] Starting epoch: '+str(i))
        np.random.shuffle(train_data)
        batches = np.array_split(train_data, 10)
        for batch in batches:
            input = []
            label = []
            for item in batch:
                input.append([item[0][np.newaxis, :],item[1][np.newaxis, :]])
                label.append(item[2][np.newaxis, :])
            model.fit(x=input, y=label, epochs=1, batch_size=len(batch))
    
    print('[INFO] Finished training')
    end = datetime.now()
    print(end-start)

    # Train the model
    print("[INFO] training model...")
    

    input = []
    label = []
    for item in valid_data:
        input.append([item[0][np.newaxis, :],item[1][np.newaxis, :]])
        label.append(item[2][np.newaxis, :])
    print('Test score:', model.evaluate(x=input, y=label, batch_size=4000, verbose=0 ))

    np.random.shuffle(test_data)
    print("[INFO] predict data...")
    for item in test_data[:10]:
        input1 = item[0]
        input2 = item[1]
        print([item[2][0], item[3]])
        prediction = model.predict([input1[np.newaxis, :], input2[np.newaxis, :]])
        prediction.append(crps.norm(item[2][:1], prediction))
        print(prediction)
    end = datetime.now()
    print(end-start)
    print('Finished')

if __name__ == "__main__":
    main(args)
