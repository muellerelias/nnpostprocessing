from tensorflow.keras.layers import Activation, Dense, Input, Concatenate, Flatten, InputLayer
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
"""
Example: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
"""


def build_multi_input_model(shape_vec, shape_mat, shape_out):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # first branch for the vector input
    inp1 = Input(shape=shape_vec)
    model1 = Dense(9, activation='linear')(inp1)
    model1 = Model(inp1, model1)

    # second branch for the matrix input
    inp2 = Input(shape=shape_mat)
    model2 = Flatten()(inp2)  # (model2)
    model2 = Dense(38, activation='linear')(model2)
    model2 = Model(inp2, model2)

    # concatenate the two inputs
    combined = Concatenate(axis=1)([model1.output, model2.output])

    # add the hiddden layers
    # x = Dense(25, activation="relu", kernel_initializer='random_normal', bias_initializer='zeros')(combined)
    x = Dense(2, activation='linear')(combined)  # (x)

    # returns the Model
    return Model([model1.input, model2.input], outputs=x)


def printModel(model):
    tf.keras.utils.plot_model(model, to_file='my_model.png', show_shapes=True,
                              show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)


def build_one_input_model(shape):
    """Build (and compile) multi input network.
    Args: 
        shape: Shape of the input vector
    Returns:
        model: Keras model
    """

    # first branch for the vector input
    inp = Input(shape=shape)
    model = Dense(28, activation='linear')(inp)
    #model = Dense(32, activation='linear')(inp)
    #model = Dense(16, activation='linear')(model)
    #model = Dense( 8, activation='linear')(model)
    #model = Dense( 4, activation='linear')(model)
    model = Dense( 2, activation='linear')(model)

    # returns the Model
    return Model(inp, model)
