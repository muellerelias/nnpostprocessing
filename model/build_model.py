from tensorflow.keras.layers import Activation, Dense, Input, Concatenate, Flatten, InputLayer
from tensorflow.keras.models import Model, Sequential

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
    inp1   = Input( shape = shape_vec )
    model1 = Dense( 9, activation="relu" )( inp1 )
    model1 = Model( inp1, model1 )


    # second branch for the matrix input
    inp2   = Input( shape=shape_mat )
    model2 = Dense( 10, activation="relu" )( inp2 )
    model2 = Dense( 5, activation="relu" )( model2 )
    model2 = Flatten()(model2)
    model2 = Model( inp2,model2 )
    
    ## concatenate the two inputs
    combined = Concatenate(axis=1)([model1.output, model2.output])
    
    ## add the hiddden layers
    x = Dense(25, activation="relu")(combined)
    x = Dense(4, activation="relu")(x)

    # returns the Model
    return Model([model1.input, model2.input], outputs=x)