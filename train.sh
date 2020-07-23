#!/bin/bash


python3 train.py \
    --name h28-e300-b512-relu \
    --data_numpy /home/elias/Nextcloud/1.Masterarbeit/Daten/vorverarbeitet2/ \
    --log_dir /home/elias/Nextcloud/1.Masterarbeit/Tests/ \
    --batch_size 512 \
    --epochs 300