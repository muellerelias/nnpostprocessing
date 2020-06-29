import model.build_model as modelprovider
from tensorflow.keras.optimizers import Adam
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

    # Plot the model
    #utils.plot_model(model, "my_model.png", show_shapes=True)

    # run the model
    prediction = []
    for item in data[:10]:
        input1 = item[0]
        input2 = item[1]
        prediction.append(model.predict([input1[np.newaxis, :], input2[np.newaxis, :]]))
    tf.nn.softmax(prediction)

    for item in prediction:
        print(item)

if __name__ == "__main__":
    main(args.filepath)
