"""
Definition of CRPS loss function.
"""
import tensorflow.keras
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf


"""
referenz: https://github.com/slerch/ppnn/blob/master/nn_postprocessing/nn_src/losses.py
"""

def crps_cost_function(y_true, y_pred):
    """Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.
    Code inspired by Kai Polsterer (HITS).
    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        theano: Set to true if using this with pure theano.
    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]

    y_true = y_true[:, 0]

    # To stop sigma from becoming negative we first have to
    # convert it the the variance and then take the square
    # root again.
    var = K.square(sigma)
    # The following three variables are just for convenience
    loc = (y_true - mu) / K.sqrt(var)
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-1*K.square(loc) / 2.0)
    Phi = 0.5 * (1.0 + tf.math.erf(loc / np.sqrt(2.0)))

    # First we will compute the crps for each input/target pair
    crps = K.sqrt(var) * (loc * (2. * Phi - 1.) +
                          2 * phi - 1. / np.sqrt(np.pi))
    # Then we take the mean. The cost is now a scalar
    return K.mean(crps)


# Version in which negative values of the standard deviation are sanctioned with the addition of an error of 10
def crps_cost_function_positiv(y_true, y_pred):
    bool = K.less(y_pred[:, 1], 0)
    result = crps_cost_function(y_true, y_pred)
    return result+K.cast(bool, dtype="float32")*10