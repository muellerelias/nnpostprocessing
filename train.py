import model.build_model as modelprovider
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import utils
import tensorflow as tf
import dataset.provider as dataset
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description='This is the inference script')

parser.add_argument("--dataset", dest="filepath", required=True,
        help="folder where the dataset is", metavar="FILE", default='/home/elias/Nextcloud/1.Masterarbeit/Daten/2020_MA_Elias/')

args = parser.parse_args()

def main(path):
    # get the data
    data = dataset.read(os.path.join(path+'ecmwf_*_240.csv'))

    # get the shape
    shape_vec, shape_mat, shape_out = dataset.shape(data[1])

    # Loading the model
    model = modelprovider.build_multi_input_model(shape_vec, shape_mat, shape_out)

    # compiling the model
    opt = SGD(lr=1e-3)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    # train the model

    # run the model

    print("[INFO] training model...")
    # Training the model
    model.fit(x=[input1,input2], y=label, epochs=1, batch_size=1)


if __name__ == "__main__":
    main(args.filepath)
