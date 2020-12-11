import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import dataset.helper.crps as crps


def mkdir_not_exists(path):
    if not os.path.exists(path):
        return os.makedirs(path)
    else:
        return True


def load_data(path, name):
    return np.load(os.path.join(path, name), allow_pickle=True)


def calculatePIT(value, loc, scale):
    return norm.cdf(value, loc=loc, scale=scale)


def datasetPIT(predictions, test_data):
    if len(predictions) != len(test_data):
        raise NameError('both sets must have the same length')
    pit = []
    for i in range(len(test_data)):
        pred = predictions[i]
        item = test_data[i]
        pit.append(calculatePIT(
            item, pred[0], abs(pred[1])))
    return pit


def printHist(data, r='', b=12):
    if r=='':
        histo = plt.hist(data, bins=b)
    else:
        histo = plt.hist(data, bins=b, range=r)
    return_string = ''
    return_array = []
    for i in range(len(histo[0])):
        return_array.append((round(histo[1][i], 2), histo[0][i]))
        return_string += '(' + \
                           str(round(histo[1][i], 2)) + \
                               ',' + str(histo[0][i])+') '
    print(return_string)
    return return_array


def printIntCountries(test_data_labels, test_data_countries, mean_predictions):
    test_crps = crps.norm_data(test_data_labels, mean_predictions)
    test_score = round(test_crps.mean(), 2)
    result = str(test_score)
    for i in [8,16,2,5,20]:
        filter = test_data_countries == i
        filter_data = test_crps[filter]
        if len(filter_data) > 0:
            item = round(np.array(filter_data).mean(), 2)
        else:
            item = 0
        result += '&{:.2f}'.format(item) 
    print(result)


def printIntMonth(test_data_labels, test_data_month, mean_predictions):
    test_crps = crps.norm_data(test_data_labels, mean_predictions)
    for i in range(1, 13):
        filter = test_data_month == i
        filter_data = test_crps[filter]
        if len(filter_data) > 0:
            item = (i, round(np.array(filter_data).mean(), 2))
        else:
            item = (i, 0)
        print(item)    