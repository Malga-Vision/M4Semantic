#!/bin/bash

savepath=$1;

python main.py --mode=train --dataset="aeroscapes" --seq_len=4 --db_seq_len=4 --arch_depth=5 --ckpt_dir="$savepath" --log_dir="$savepath/summaries" --records=data/aeroscapes/train_data/ --enable_validation $2
