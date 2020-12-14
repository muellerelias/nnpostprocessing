# go to parent directory
import sys
sys.path.append('..')

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import helper as helpers


import dataset.converter as converter
import math



def main(size=2500):
    start = datetime.now()
    ud = []
    nd = []
    od = []
    for i in range(size):
        y = np.random.normal(1, 1, size=1)[0]
        ud.append(helpers.calculatePIT(y, 1,  9/16 ) )
        nd.append(helpers.calculatePIT(y, 1,     1 ) )
        od.append(helpers.calculatePIT(y, 1, 25/16 ) )


    printHist(ud)
    printHist(nd)
    printHist(od)

def printHist(data):
    histo = plt.hist(data, bins=20, range=(0, 1))
    return_string = ''
    return_array  = []
    for i in range(len(histo[0])):
        return_array.append((round(histo[1][i],2),histo[0][i]))
        return_string += '('+str(round(histo[1][i],2))+','+ str(histo[0][i])+') '
    print(return_string)
    return return_array

if __name__ == "__main__":
    main()
