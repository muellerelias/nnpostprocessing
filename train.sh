#!/bin/bash


python3 train.py \
    --name normalised-multi-b800-e40 \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitetNorm/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 800 \
    --epochs 150 \
    --model multi