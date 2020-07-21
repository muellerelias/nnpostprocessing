#!/bin/bash


python3 train.py \
    --name 28-hidden-e50-b1000 \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitet2/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 1000 \
    --epochs 50