#!/bin/bash

python3 inference.py \
  --hub_id ARG-NCTU \
  --repo_id detr-resnet-50-finetuned-600-epochs-GuardBoat-dataset \
  --input_path source_videos/Multi_Boat.mp4 \
  --output_path output_videos/Multi_Boat.mp4 \
  --confidence_threshold 0.5

