#!/bin/bash

pip3 install -r requirements.txt

python3 show_batch.py --batch_size=$1 --file_path=$2
