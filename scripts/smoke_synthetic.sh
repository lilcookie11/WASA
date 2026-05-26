#!/bin/bash
set -e
python3 tools/make_synthetic_psm.py --output dataset/synthetic_psm --length 192 --dims 4
python3 -u main_infoflow_standalone.py \
  --mode train_test \
  --dataset PSM \
  --data_path ./dataset/synthetic_psm \
  --input_c 4 \
  --output_c 4 \
  --win_size 16 \
  --step 16 \
  --batch_size 4 \
  --num_epochs 1 \
  --patience 1 \
  --d_model 32 \
  --n_heads 4 \
  --e_layers 1 \
  --d_ff 64 \
  --bottleneck_dim 12 \
  --flow_layers 2 \
  --threshold_mode paper
