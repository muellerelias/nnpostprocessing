# Using weather regime information in post-processing of country-aggregated medium-range weather forecasts

The chaotic nature of the atmosphere limits predictability of local weather to a few days. Still, on longer time scales large-scale circulation patterns describe the variability of weather characteristics over larger regions. This provides a forecast opportunity on the so-called subseasonal time scale of several days to a few weeks (10-60 days).
In this project we aim to elucidate if there is an additional value in using forecast information about large-scale circulation patterns in the neural network-based post processing model of Rasp and Lerch (2018).  

## Dataset



## Requirements

- python 3
- pip3

## Installation

```bash
pip3 install tensorflow
```

## Usage
- Change the Path variable in dataset/read_csv_files.py
- For training a model use ```bash python3 train.py```