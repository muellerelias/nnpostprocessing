#!/bin/bash


python3 train.py \
    --exp_name versuch-1-3 \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/10days/vorverarbeitetRegime/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 16 \
    --epochs 30 \
    --initial_epochs 0 \
    --learning_rate 5e-05