#!/bin/bash

while true; do
    python3 train.py \
    --save_model_hub_id ARG-NCTU \
    --save_model_repo_id detr-resnet-50-finetuned-600-epochs-TW-Marine-5cls-dataset \
    --load_model_hub_id ARG-NCTU \
    --load_model_repo_id detr-resnet-50-finetuned-600-epochs-TW-Marine-2cls-dataset \
    --dataset_hub_id ARG-NCTU \
    --dataset_repo_id TW_Marine_5cls_dataset \
    --dataset_format parquet \
    --epoch 450 \
    --batch_size 2 \
    --learning_rate 1e-5 \
    --weight_decay 1e-4 \
    --logging_steps 50 \
    --save_total_limit 5 \
    --classes_path data/TW_Marine_5cls_classes.txt \
    --image_height 480 \
    --image_width 1920 \
    --device cuda
done