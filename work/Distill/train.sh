#!/bin/bash

set -e

output=logdir
mkdir -p $output
echo "makedir" $output
# 3个命令串行
python main.py -c configs/self_distill.yaml > $output/self_distill.log 2>&1
python main.py -c configs/self_distill2.yaml -w output/self_distill/nwc5-1-filtered_epoch_00120.pdparams > $output/self_distill2.log 2>&1
python main.py -c configs/self_distill_confusion.yaml -w output/self_distill2/nwc5-1-filtered_epoch_00120.pdparams > $output/self_distill_confusion.log 2>&1
