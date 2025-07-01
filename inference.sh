#!/bin/bash

python3 inference.py \
  --hub_id ARG-NCTU \
  --repo_id detr-resnet-50-finetuned-600-epochs-TW-Marine-5cls-dataset \
  --input_path source_videos/S1_ch1234_20250610_1255.mp4 \
  --output_path output_videos/S1_ch1234_20250610_1255_output.mp4 \
  --confidence_threshold 0.5

