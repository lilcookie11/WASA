#!/bin/bash
python3 -u main_anomaly_infoflow.py \
  --mode train_test \
  --dataset PSM \
  --data_path ./dataset/PSM \
  --input_c 25 \
  --output_c 25 \
  --win_size 100 \
  --step 1 \
  --batch_size 256 \
  --num_epochs 3 \
  --model_save_path checkpoints_anomaly_infoflow_psm \
  --result_path results/anomaly_infoflow_psm \
  --beta 0.6 \
  --gamma 0.4 \
  --alpha 0.5 \
  --association_weight 1.0 \
  --info_train_weight 0.1 \
  --threshold_mode percentile \
  --anormly_ratio 0.75
