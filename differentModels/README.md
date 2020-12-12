### Different models

This folder contains a Python script for each model, which trains 10 versions of it.
Please modify the following variables before running the script:
- expname: Is the name of the model
- forecast: Is the forecast horizon
- numpy_path: is the absolute path to the folder, which contains the numpy files
- logdir: Is the folder where you save all experiments/models
- batchsize: Is the batchsize of your training. I used 16.
- epochs: Is the amaount of epochs for training. I used 30.
- initial_epochs: Is the number in which epoch the training start. I used 0.
- learning_rate: Is the leanring rate for training. I used 5e-05.
- train_model: Is a boolean, If true: you train a model and evaluete it, if False: you only evaluate it.
