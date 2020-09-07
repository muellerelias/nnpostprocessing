#!/bin/bash


python3 train.py \
    --exp_name ganzesNetz3hidden_with_Date \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 1 \
    --epochs 50 \
    --initial_epochs 0 \
    --learning_rate 0.10657513276093987 \
    --model multi