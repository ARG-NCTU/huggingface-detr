#!/bin/bash

python3 eval.py \
  --hub_id ARG-NCTU \
  --repo_id detr-resnet-50-finetuned-600-epochs-GuardBoat-dataset \
  --dataset_hub_id ARG-NCTU \
  --dataset_repo_id GuardBoat_dataset_2025 \
  --dataset_format parquet \
  --classes_path data/GuardBoat_classes.txt \
  --image_height 480 \
  --image_width 1920 \
  --batch_size 8 \
  --num_workers 4 \
  --device cuda
