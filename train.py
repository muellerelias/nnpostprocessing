import model.build_model as modelprovider
import model.loss_functions as loss
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import utils
import tensorflow as tf
import dataset.provider as dataset
import numpy as np
import argparse
import os

parser = argparse.ArgumentParser(description='This is the inference script')

dataarg = parser.add_mutually_exclusive_group( required=True)
dataarg.add_argument("--dataset", dest="filepath",
        help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

dataarg.add_argument("--numpyfile", dest="numpyfile",
        help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

parser.add_argument("--model", dest="modelpath",
        help="path to model, from where it should load, or where it should save it", metavar="FILE")

args = parser.parse_args()

def main(path=None, numpy=None, modelpath=None):

    # get the data
    if (path):
        data = dataset.read(os.path.join(path+'ecmwf_*_240.csv'))

    # get the shape
    shape_vec, shape_mat, shape_out = dataset.shape(data[1])

    """
    if (modelpath):
        model = tf.keras.models.load_model(modelpath)
    else:
    """
    # Loading the model
    model = modelprovider.build_multi_input_model(shape_vec, shape_mat, shape_out)
    modelprovider.printModel(model)

    # compiling the model


    # Training the model
    input = []
    label = []
    for item in data:
        input.append([item[0][np.newaxis, :],item[1][np.newaxis, :]])
        label.append(item[2][np.newaxis, :])
    
    # Train the model
    print("[INFO] training model...")
    
    lossfn = loss.crps_cost_function 
    lr = 0.01
    opt = SGD(lr=lr)
    model.compile(loss = lossfn, optimizer = opt)
    model.fit(x=input, y=label, epochs=1, batch_size=5)
    
    print('Test score:', model.evaluate(x=input, y=label, batch_size=4000, verbose=0 ))

    #model.save(modelpath)

    print("[INFO] predict data...")
    for item in data[50:55]:
        input1 = item[0]
        input2 = item[1]
        print(item[2][:1])
        prediction = model.predict([input1[np.newaxis, :], input2[np.newaxis, :]])
        tf.nn.softmax(prediction)
        print(prediction)

if __name__ == "__main__":
    main(args.filepath, modelpath=args.modelpath)
