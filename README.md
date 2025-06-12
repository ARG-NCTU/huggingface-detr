# HuggingFace DETR

HuggingFace DETR Training, Evaluation, Inferencing Guide

## Clone repo 

```
git clone git@github.com:ARG-NCTU/huggingface-detr.git
``` 

## Enter the repo

```bash
cd ~/huggingface-detr/
```

## Enter Docker Environment

For first terminal:

```bash
source Docker/ros1-gpu/run.sh
```

More terminal:

```bash
source Docker/ros1-gpu/join.sh
```

## Prepare Dataset

Download HuggingFace dataset:

```bash
huggingface-cli login
```
```bash
huggingface-cli download ARG-NCTU/Boat_dataset_2024 \
--repo-type dataset \
--local-dir ~/huggingface-detr
```

Unzip images:

```bash
unzip ~/huggingface-detr/data/images.zip -d ~/huggingface-detr/
```

## Training, Evaluation, Inferencing

Log in huggingface

```bash
huggingface-cli login
```

Enter the repo

```bash
cd ~/huggingface-detr/
```

### Training

```bash
python3 train.py \
--save_model_hub_id ARG-NCTU \
--save_model_repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset \
--load_model_hub_id facebook \
--load_model_repo_id detr-resnet-50 \
--dataset_hub_id ARG-NCTU \
--dataset_repo_id Boat_dataset_2024 \
--dataset_format jsonl \
--epoch 20 \
--batch_size 8 \
--learning_rate 1e-5 \
--weight_decay 1e-4 \
--logging_steps 50 \
--save_total_limit 100 \
--classes_path data/classes.txt \
--image_height 480 \
--image_width 640 \
--device cuda
```

Or modify the train.sh and run it:

```bash
source train.sh
```

Upload model weights to hub (If push_to_hub not working)

```bash
huggingface-cli upload ARG-NCTU/detr-resnet-50-finetuned-20-epochs-Boat-dataset \
detr-resnet-50-finetuned-20-epochs-Boat-dataset \
--repo-type=model \
--commit-message="Upload model weights to hub"
```

Use tensorboard to see training logs

```bash
tensorboard --logdir=detr-resnet-50-finetuned-20-epochs-Boat-dataset/runs
```

### Evaluation

```bash
python3 eval.py \
--hub_id ARG-NCTU \
--repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset \
--dataset_hub_id ARG-NCTU \
--dataset_repo_id Boat_dataset_2024 \
--dataset_format jsonl \
--classes_path data/classes.txt \
--image_height 480 \
--image_width 640 \
--batch_size 8 \
--num_workers 4 \
--device cuda
```

Or modify the eval.sh and run it:

```bash
source eval.sh
```

### Inferencing

Download Source Videos [Link](http://gofile.me/773h8/baC1yKEOm)

```bash
python3 inference.py \
--hub_id ARG-NCTU \
--repo_id detr-resnet-50-finetuned-20-epochs-Boat-dataset \
--input_path source_videos/Multi_Boat.mp4 \
--output_path output_videos/Multi_Boat.mp4 \
--confidence_threshold 0.5
```

### Build ROS1 Workspace

Enter the repo

```bash
cd ~/huggingface-detr/
```

Setup ROS

```bash
source environment_ros1.sh 
```

Clean catkin ws

```bash
source clean_ros1_all.sh
```

Build catkin ws

```bash
source build_ros1_all.sh
```

### ROS1 Inference

Setup ROS

```bash
source environment_ros1.sh 
```

Download DETR model from huggingface hub

```bash
roslaunch detr_inference download_model.launch
```

ROS1 Inference local DETR model

```bash
roslaunch detr_inference detr_inference.launch 
```

Or ROS1 Inference local DETR model for visual servoing

```bash
roslaunch detr_inference detr_inference_searching.launch 
```

Or ROS1 Inference local DETR model for distance & angle detection

```bash
roslaunch detr_inference detr_inference_searching_distance.launch 
```
