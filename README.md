# Using weather regime information in post-processing of country-aggregated medium-range weather forecasts


This is the git-repo to my master thesis with the topic: "Using weather regime information in post-processing of country-aggregated medium-range weather forecast"

## Dataset
The following data were used:
- sub-seasonal to seasonal ensemble dataset from ECMWF. which are country-aggregated
- Atlantic-European weather regime projection's, calculated with dhe sub-seasonal to seasonal dateset


## Requirements

- python 3
- pip3

## Installation

```bash
pip3 install tensorflow properscoring matplotlib pandas keras-tuner
```

## Usage
- To convert a data set consisting of CSV files into a Numpy Array, which is used in training. Please modify and execute the script [build_np_array](../blob/master/build_np_array). 
- The directory [differentModels](../blob/master/differentModels) contains 10 different Python files for training the 10 models shown in Subsection 4.3 of my master thesis.

## Acknowledge
I was privileged to write a very interesting thesis, for which I would like to thank two groups who made this thesis possible:
- I thank [Prof. Dr. Tilmann Gneiting](https://www.math.kit.edu/stoch/~gneiting/en) for supervising this thesis and [Dr. Sebastian Lerch](https://www.math.kit.edu/stoch/~lerch/de) for inspiring conversations and constructive remarks.
- I thank [Dr. Christian Grams'](https://www.imk-tro.kit.edu/14_7356.php) working group for the opportunity and support for this thesis. Especially [Dr. Julian Quinting](https://www.imk-tro.kit.edu/14_7532.php) for his expert support and [Dr. Dominik Büeler](https://www.imk-tro.kit.edu/14_7600.php) for the calculation of the data. 
