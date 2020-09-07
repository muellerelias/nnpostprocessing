#!/bin/bash


python3 train_country.py \
    --exp_name UK-multi-1-more-data \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 10 \
    --epochs 50 \
    --model multi