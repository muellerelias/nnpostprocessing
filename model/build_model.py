from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.models import Model, Sequential, Concatenate


"""
Example: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
"""

def create_vec_input(dim):
    """Build the input Branche for the vector
    Args:
        dim: the dimension for the input vector
    """

    # define our MLP network
    model = Sequential()
    model.add(Dense(20, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
	# return our model
    return model

def create_mat_input(inp_shape):
    # initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering

	# define the model input
    inputs = Input(shape=inp_shape)
    model = Sequential()
    model.add(Dense(10, input_dim=inputs , activation="relu"))
    model.add(Dense(20, activation="relu"))
    return model

def build_multi_input_model(shape_vec, shape_mat, shape_out, compile=False, optimizer='adam',
                   lr=0.05, loss=crps_cost_function):
    """Build (and compile) multi input network.
    Args:
        shape_vec: Shape of the input vector
        shape_mat: Shape of the input matrix
        compile: If true, compile model
        optimizer: Name of optimizer
        lr: learning rate
        loss: loss function
    Returns:
        model: Keras model
    """

    # create the two inputs
    inp1 = create_vec_input(shape_vec)
    inp2 = create_mat_input(shape_mat)
    
    # concatenate the two inputs
    combinedInput = Concatenate([inp1.output, inp2.output])

    # add the hiddden layers
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    # our final model will accept categorical/numerical data on the MLP
    # input and images on the CNN input, outputting a single value (the
    # predicted price of the house)
    model = Model(inputs=[inp1.input, inp2.input], outputs=x)
    if compile:
        opt = keras.optimizers.__dict__[optimizer](lr=lr)
        model.compile(optimizer=opt, loss=loss)
    return model
