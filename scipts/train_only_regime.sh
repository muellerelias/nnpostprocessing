#!/bin/bash


python3 train_only_regime.py \
    --exp_name ganzesNetzOnlyDate \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 1 \
    --epochs 30 \
    --initial_epochs 10 \
    --learning_rate 0.0004393311274534397 \
    --model multi