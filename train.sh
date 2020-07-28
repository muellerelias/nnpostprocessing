#!/bin/bash


python3 train.py \
    --name normalised-multi-b1024-e30 \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 1024 \
    --epochs 30 \
    --model multi