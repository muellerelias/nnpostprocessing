from tensorflow.keras.layers import Activation, Dense, Input, Concatenate, Flatten, InputLayer
from tensorflow.keras.models import Model, Sequential

"""
Example: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
"""

def create_vec_input(inp_shape):
    """Build the input Branche for the vector
    Args:
        inp_shape: the dimension for the input vector
    """
    # define our MLP network
    model = Sequential()
    model.add(Dense(9, input_shape=inp_shape, activation="relu"))
	# return our model
    return model

def create_mat_input(inp_shape):
    # initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	# define the model input
    #model = Sequential()
    inp = Input(shape=inp_shape)
    model = Dense(19, activation="relu")(inp)
    model = Dense(10, activation="relu")(model)
    model = Flatten()(model)
    return Model(inp, model)

def build_multi_input_model_old(shape_vec, shape_mat, shape_out):
    """Build (and compile) multi input network.
    Args:
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
    Returns:
        model: Keras model
    """

    # create the two inputs
    inp1 = create_vec_input(shape_vec)
    inp2 = create_mat_input(shape_mat)
    
    # concatenate the two inputs
    combinedInput = Concatenate(axis=1)([inp1.output, inp2.output])

    # add the hiddden layers
    x = Dense(25, activation="relu")(combinedInput)
    # x = Dense(15, activation="relu")(x)
    # x = Dense(10, activation="relu")(x)
    x = Dense(4, activation="relu")(x)

    # returns the Model
    return Model(inputs=[inp1.input, inp2.input], outputs=x)

def build_multi_input_model(shape_vec, shape_mat, shape_out):
    """Build (and compile) multi input network.
    Args: 
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
    Returns:
        model: Keras model
    """
    
    inp1   = Input( shape = shape_vec )
    model1 = Dense( 9, activation="relu" )( inp1 )
    model1 = Model( inp1, model1 )

    inp2   = Input( shape=shape_mat )
    model2 = Dense( 10, activation="relu" )( inp2 )
    model2 = Dense( 5, activation="relu" )( model2 )
    model2 = Flatten()(model2)
    model2 = Model( inp2,model2 )
    
    ## concatenate the two inputs
    combined = Concatenate(axis=1)([model1.output, model2.output])
    
    ## add the hiddden layers
    x = Dense(25, activation="relu")(combined)
    ## x = Dense(15, activation="relu")(x)
    ## x = Dense(10, activation="relu")(x)
    x = Dense(4, activation="relu")(x)

    # returns the Model
    #return Model(inputs=[model2.input, model2.input], outputs=[model2.output , model2.output])
    return Model([model1.input, model2.input], outputs=x)