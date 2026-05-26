#!/bin/bash
python3 -u main_infoflow_standalone.py \
  --mode train_test \
  --dataset MSL \
  --data_path ./dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --win_size 100 \
  --step 100 \
  --batch_size 32 \
  --num_epochs 10 \
  --beta 0.6 \
  --gamma 0.4 \
  --alpha 0.5 \
  --threshold_mode paper
