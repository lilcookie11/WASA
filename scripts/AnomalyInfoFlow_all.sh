#!/bin/bash
set -e

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

python3 -u main_anomaly_infoflow.py \
  --seed 7 \
  --mode train_test \
  --dataset MSL \
  --data_path ./dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --win_size 100 \
  --step 1 \
  --batch_size 256 \
  --num_epochs 3 \
  --model_save_path checkpoints_anomaly_infoflow_msl \
  --result_path results/anomaly_infoflow_msl \
  --beta 0.6 \
  --gamma 0.4 \
  --alpha 0.5 \
  --association_weight 1.0 \
  --info_train_weight 0.1 \
  --threshold_mode percentile \
  --anormly_ratio 1.0

python3 -u main_anomaly_infoflow.py \
  --mode train_test \
  --dataset SMD \
  --data_path ./dataset/SMD \
  --input_c 38 \
  --output_c 38 \
  --win_size 100 \
  --step 100 \
  --batch_size 256 \
  --num_epochs 10 \
  --model_save_path checkpoints_anomaly_infoflow_smd \
  --result_path results/anomaly_infoflow_smd \
  --weight_decay 0.0 \
  --beta 0.6 \
  --gamma 0.4 \
  --alpha 0.5 \
  --association_weight 1.0 \
  --info_train_weight 0.0 \
  --threshold_mode percentile \
  --anormly_ratio 0.35
