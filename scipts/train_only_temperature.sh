#!/bin/bash


python3 train_only_temperatur.py \
    --exp_name ganzesNetzOnlyTemperature5 \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 1 \
    --epochs 38 \
    --initial_epochs 0 \
    --learning_rate 0.00706622494565155 \
    --model multi