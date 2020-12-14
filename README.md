# Using weather regime information in post-processing of country-aggregated medium-range weather forecasts


This is the git-repo to my master thesis with the topic: "Using weather regime information in post-processing of country-aggregated medium-range weather forecast"

## Dataset
The following data are used:
- sub-seasonal to seasonal ensemble dataset from ECMWF, which are country-aggregated
- Atlantic-European weather regime projections, calculated with the sub-seasonal to seasonal data set


## Requirements

- python 3
- pip3

## Installation

```bash
pip3 install tensorflow properscoring matplotlib pandas keras-tuner
```

## Usage
- To convert a data set consisting of CSV files into a Numpy Array, which is used in training. Please modify and execute the script [build_np_array](build_np_array.sh). 
- The directory [differentModels](/differentModels) contains 10 different Python files for training, the 10 models shown in Subsection 4.3 of my master thesis.
- For training the model with all inputs, edit and execute [train.sh](/train.sh)
- The directory [hyperParameterSearch](/hyperParameterSearch) contains different Hyper-Parameter search files
- The directory [scipts](/scipts) contains different single script files:
  - The script [climatology](/scipts/climatology.py) calculates the climatology
  - The script [ensemble CRPS](/scripts/ensemble_CRPS.py) returns the ensemble CRPS of the data sets
  - The script [feature](/scripts/feature.py) computes the feature importance
  - The script [PITHist](/scripts/PITHist.py) calculates the PIT histograms of example in Section 2.2
  - The script [test CRPS](/scripts/test_CRPS.py) returns the ensemble CRPS of test set, filtered after month and countries


## Acknowledgement
I was privileged to write a very interesting thesis, for which I would like to thank two groups who made this thesis possible:
- [Prof. Dr. Tilmann Gneiting](https://www.math.kit.edu/stoch/~gneiting/en) for supervising this thesis and [Dr. Sebastian Lerch](https://www.math.kit.edu/stoch/~lerch/de) for supervising this thesis and [Dr. Sebastian Lerch](https://www.math.kit.edu/stoch/~lerch/de) for his inspiring conversations and constructive remarks.
- The working group (Großräumige Dynamik und Vorhersagbarkeit) of [Dr. Christian Grams'](https://www.imk-tro.kit.edu/14_7356.php), for meteorological support. Especially Dr. Julian Quinting for the scientific exchange and [Dr. Dominik Büeler](https://www.imk-tro.kit.edu/14_7600.php) for the calculation of the data sets.
