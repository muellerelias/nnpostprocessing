import numpy as np


def verificationRank(value, data):
    data  = np.append( data , np.array( [ value ] ) , axis=0)
    ranks = np.empty_like( data )
    ranks[ np.argsort( data ) ] = np.arange( len( data ) )
    return ranks[-1]+1

if __name__ == "__main__":
    result = verificationRank(12, [1,2,3,4,5,6,7,8,9,10,11])
    print(result)