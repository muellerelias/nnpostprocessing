from tensorflow.keras.layers import Activation, Dense, Input, Concatenate, Flatten, InputLayer, Embedding
from tensorflow.keras.models import Model, Sequential
import tensorflow as tf
import os



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
    model1 = Embedding(23, 2, name='Country_Embedding')(inp1)
    model1 = Flatten()(model1)

    # second branch for the vector input
    inp2 = Input(shape=shape_vec, name="Date_and_Regimes")

    # third branch for the matrix input
    inp3 = Input(shape=shape_mat, name="Ensemble")
    model3 = Flatten()(inp3)
    
    # concatenate the two inputs
    x = Concatenate(axis=1)([model1, inp2, model3])

    # add the hiddden layers
    x = Dense( 100 , activation='linear' , name="Combined_Hidden_Layer_1" )( x )
    x = Dense( 100 , activation='relu'   , name="Combined_Hidden_Layer_2" )( x )
    x = Dense( 100 , activation='relu'   , name="Combined_Hidden_Layer_3" )( x )

    x = Dense(   2 , activation='linear' , name="Output_Layer" )(x)

    # returns the Model
    return Model([inp1, inp2, inp3], outputs=x)


def printModel(model, dir='', name='my_model.png'):
    tf.keras.utils.plot_model(model, to_file=os.path.join(dir , name), show_shapes=True,
                              show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96)

def reset_weights(model):
    for layer in model.layers: 
        if hasattr(layer,'init'):
                input_dim = layer.input_shape[1]
                new_weights = layer.init((input_dim, layer.output_dim),name='{}_W'.format(layer.name))
                layer.trainable_weights[0].set_value(new_weights.get_value())