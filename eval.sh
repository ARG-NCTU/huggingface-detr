#!/bin/bash

python3 eval.py \
    --model_type detr \
    --hub_id ARG-NCTU \
    --repo_id detr-resnet-50-finetuned-600-epochs-TW-Marine-5cls-dataset \
    --dataset_hub_id ARG-NCTU \
    --dataset_repo_id TW_Marine_5cls_dataset \
    --dataset_choice test \
    --dataset_format parquet \
    --classes_path data/TW_Marine_5cls_classes.txt \
    --image_height 480 \
    --image_width 1920 \
    --batch_size 2 \
    --num_workers 2 \
    --device cuda
