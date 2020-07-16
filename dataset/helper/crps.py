import numpy as np
import properscoring as ps

def ensemble(label, ensemble):
    return ps.crps_ensemble(label, ensemble)

def norm(label, pred):
    return ps.crps_gaussian(label, pred[0], pred[1])