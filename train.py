import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow.keras.optimizers import SGD, Adam

import dataset.provider as dataset
import model.build_model as modelprovider
import model.loss_functions as loss

parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--data_numpy", dest="numpy_path",
        help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

args = parser.parse_args()

def main(args):
    
    # get the data
    train_data = np.load(os.path.join(args.numpy_path, 'train_set.npy'), allow_pickle=True)
    valid_data = np.load(os.path.join(args.numpy_path, 'valid_set.npy'), allow_pickle=True)
    test_data = np.load(os.path.join(args.numpy_path, 'test_set.npy'), allow_pickle=True)
    
    # get the shape
    shape_vec, shape_mat, shape_out = dataset.shape(train_data[1])

    # Loading the model
    model = modelprovider.build_multi_input_model(shape_vec, shape_mat, shape_out)
    modelprovider.printModel(model)

    # compiling the model
    lossfn = loss.crps_cost_function 
    lr = 0.01
    opt = SGD(lr=lr)
    model.compile(loss = lossfn, optimizer = opt)

    # Training the model
    input = []
    label = []
    for item in train_data:
        input.append([item[0][np.newaxis, :],item[1][np.newaxis, :]])
        label.append(item[2][np.newaxis, :])
    
    # Train the model
    print("[INFO] training model...")
    model.fit(x=input, y=label, epochs=5, batch_size=500)
    
    #print('Test score:', model.evaluate(x=input, y=label, batch_size=4000, verbose=0 ))

    print("[INFO] predict data...")
    for item in valid_data[50:55]:
        input1 = item[0]
        input2 = item[1]
        print(item[2][:1])
        prediction = model.predict([input1[np.newaxis, :], input2[np.newaxis, :]])
        print(prediction)

if __name__ == "__main__":
    main(args)
