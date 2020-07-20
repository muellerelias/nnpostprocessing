#!/bin/bash


python3 train.py \
    --name Test3 \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitet2/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 50000 \
    --epochs 50