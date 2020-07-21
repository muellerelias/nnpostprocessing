import os
import numpy as np

def mkdir_not_exists(path):
    if not os.path.exists(path):
        return os.makedirs(path)
    else:
        return True

def load_data(path, name):
    return np.load(os.path.join(path, name), allow_pickle=True)
