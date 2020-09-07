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
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 20, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)
    model1 = Dense(2, activation="linear",
                   name="Embedding_Hidden_Layer")(model1)

    # second branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    model2 = Dense(30, activation="linear",
                   name="Date_and_Regimes_Hidden_Layer_1")(inp2)
    model2 = Dense(33, activation="linear",
                   name="Date_and_Regimes_Hidden_Layer_2")(model2)
    
    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    model3 = Dense(43, activation="linear",
                   name="Ensemble_Hidden_Layer_1")(model3)
    model3 = Dense(90, activation="linear",
                   name="Ensemble_Hidden_Layer_2")(model3)
    
    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, model2, model3])

    # add the hiddden layers
    x = Dense( 48, activation='linear', name="Combined_Hidden_Layer_1")(x)
    x = Dense( 91, activation='linear', name="Combined_Hidden_Layer_2")(x)
    x = Dense( 62, activation='linear', name="Combined_Hidden_Layer_3")(x)#

    x = Dense(2, activation='linear', name="Output_Layer")(x)

    # returns the Model
    return Model([inp1, inp2, inp3], outputs=x)


def printModel(model, name='my_model.png'):
    tf.keras.utils.plot_model(model, to_file=name, show_shapes=True,
                              show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)


def build_multi_country_model(shape_vec, shape_mat):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # econd branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    model2 = Dense(24, activation='linear',  name="Vector_Hidden_Layer")(inp2)

    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    model3 = Dense(28, activation='linear',
                   name="Ensemble_Hidden_Layer")(model3)

    # concatenate the two inputs
    x = Concatenate(axis=1)([model2, model3])

    # add the hiddden layers
    x = Dense(124, activation='linear', name="Combined_Hidden_Layer_1")(x)

    x = Dense(2, activation='linear', name="Output_Layer")(x)

    # returns the Model
    return Model([inp2, inp3], outputs=x)


def build_multi_big_input_model(shape_vec, shape_mat):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # first branch for the
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 9, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)

    # s
    # econd branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    model2 = Dense(42, activation="linear",
                   name="Date_and_Regimes_Hidden_Layer")(inp2)

    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    model3 = Dense(90, activation="linear",
                   name="Ensemble_Hidden_Layer")(model3)

    comb = Concatenate(axis=1)([model1, model2])
    comb = Dense(47, activation='linear',
                 name="Hidden_Date_and_Regime_combined_Hidden_Layer")(comb)

    # concatenate the two inputs
    x = Concatenate(axis=1)([comb, model3])

    # add the hiddden layers
    x = Dense(86, activation='linear', name="Combined_Hidden_Layer_1")(x)
    x = Dense(145, activation='linear', name="Combined_Hidden_Layer_2")(x)

    x = Dense(2, activation='linear', name="Output_Layer")(x)

    # returns the Model
    return Model([inp1, inp2, inp3], outputs=x)


def build_multi_input_model_no_regimes(shape_vec, shape_mat):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # first branch for the
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 8, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)

    # econd branch for the vector input
    inp2 = Input(shape=(1,), name="Date_and_Regimes")
    #model2 = Flatten()(inp2)

    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    model3 = Dense(1, activation="linear",
                   name="Ensemble_Hidden_Layer")(model3)

    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, inp2, model3])

    # add the hiddden layers
    # x = Dense(  77, activation='linear', name="Combined_Hidden_Layer_1")(x)
    x = Dense(123, activation='linear', name="Combined_Hidden_Layer")(x)

    x = Dense(2, activation='linear', name="Output_Layer")(x)
    return Model([inp1, inp2, inp3], outputs=x)


def build_multi_only_regime_model(shape_vec, shape_mat):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # first branch for the
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 11, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)
    #model1 = Dense(2, activation="linear",
    #               name="Embedding_Hidden_Layer")(model1)

    # second branch for the vector input
    inp2 = Input(shape=(1,), name="Date_and_Regimes")
    #model2 = Dense(40, activation="linear",
    #               name="Date_and_Regimes_Hidden_Layer_1")(inp2)
    #model2 = Dense(33, activation="linear",
    #               name="Date_and_Regimes_Hidden_Layer_2")(model2)
    
    # third branch for the matrix input
    #inp3 = Input(shape=shape_mat, name="Ensemble")
    #model3 = Flatten()(inp3)
    #model3 = Dense(43, activation="linear",
    #               name="Ensemble_Hidden_Layer_1")(model3)
    #model3 = Dense(90, activation="linear",
    #               name="Ensemble_Hidden_Layer_2")(model3)
    
    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, inp2])

    # add the hiddden layers
    x = Dense( 142, activation='linear', name="Combined_Hidden_Layer_1")(x)
    #x = Dense( 91, activation='linear', name="Combined_Hidden_Layer_2")(x)
    #x = Dense( 62, activation='linear', name="Combined_Hidden_Layer_3")(x)

    x = Dense(2, activation='linear', name="Output_Layer")(x)

    # returns the Model
    return Model([inp1, inp2], outputs=x)


def build_multi_input_model_only_temperature(shape_vec, shape_mat):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        shape_out: Shape of the output vector
    Returns:
        model: Keras model
    """

    # first branch for the
    inp1 = Input(shape=(1,), name='Country_ID')
    model1 = Embedding(24, 21, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)


    # second branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")
    #model2 = Dense(28, activation="linear",
    #               name="Date_and_Regimes_Hidden_Layer_1")(inp2)
    #model2 = Dense(33, activation="linear",
    #               name="Date_and_Regimes_Hidden_Layer_2")(model2)
    
    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    #model3 = Dense(40, activation="linear",
    #               name="Ensemble_Hidden_Layer_1")(model3)
    #model3 = Dense(90, activation="linear",
    #               name="Ensemble_Hidden_Layer_2")(model3)
    
    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, inp2, model3])

    # add the hiddden layers
    x = Dense( 69, activation='linear', name="Combined_Hidden_Layer_1")(x)
    x = Dense(119, activation='linear', name="Combined_Hidden_Layer_2")(x)

    x = Dense(2, activation='linear', name="Output_Layer")(x)

    # returns the Model
    return Model([inp1, inp2, inp3], outputs=x)