from tensorflow.keras.layers import Activation, Dense, Input, Concatenate, Flatten, InputLayer, Embedding
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
"""
Example: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
"""


def build_multi_input_model(shape_vec, shape_mat):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # first branch for the 
    inp1 = Input(shape=(1,))
    model1 = Embedding(24, 2)(inp1)
    model1 = Flatten()(model1)

    # s
    # econd branch for the vector input
    inp2 = Input(shape=shape_vec)
    model2 = Dense(9, activation='linear')(inp2)

    # third branch for the matrix input
    inp3 = Input(shape=shape_mat)
    model3 = Flatten()(inp3)

    # concatenate the two inputs
    combined = Concatenate(axis=1)([model1, model2, model3])

    # add the hiddden layers
    x = Dense(2, activation='linear')(combined)  # (x)

    # returns the Model
    return Model([inp1, inp2, inp3], outputs=x)


def printModel(model, name='my_model.png'):
    tf.keras.utils.plot_model(model, to_file=name, show_shapes=True,
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
    model = Dense(2, activation='linear')(model)

    # returns the Model
    return Model(inp, model)


def build_emb_model(shape):
    # first branch for the features input
    feature_in = Input(shape=(46,))
    #feature = Dense(28,activation='linear')(feature_in)

    id_in = Input(shape=(1,))
    emb = Embedding(24, 2)(id_in)
    emb = Flatten()(emb)
    
    combined = Concatenate()([feature_in, emb])
    #model = Dense(28, activation='linear')(combined)
    model = Dense(2, activation='linear')(combined)#(model)

    # returns the Model
    return Model(inputs=[id_in, feature_in], outputs=model)
