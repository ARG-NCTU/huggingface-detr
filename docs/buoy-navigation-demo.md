# 1. In perception-fusion repo:

## Clone Repo

```
git clone --recursive git@github.com:ARG-NCTU/perception-fusion.git
``` 

## Update repo and submodules

```bash
cd ~/perception-fusion
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

## Setup

```bash
cd ~/perception-fusion
source Docker/ros1-cpu/run.sh
source environment_ros1.sh
source clean_ros1_ws.sh
source build_ros1_all.sh
source environment_ros1.sh
```

## Demo

Terminal 1: Camera

```bash
cd ~/perception-fusion
source Docker/ros1-cpu/run.sh
source environment_ros1.sh
roslaunch image_processing side_camera_3.launch
```

Terminal 2: Stitching

```bash
cd ~/perception-fusion
source Docker/ros1-cpu/join.sh
source environment_ros1.sh
roslaunch image_processing image_stitcher_queue.launch
```

Terminal 3: Scaling

```bash
cd ~/perception-fusion
source Docker/ros1-cpu/join.sh
source environment_ros1.sh
roslaunch image_processing image_scale.launch
```

# 2. In huggingface-detr repo:

## Clone repo 

```
git clone git@github.com:ARG-NCTU/huggingface-detr.git
```

## Update repo and submodules

```bash
cd ~/huggingface-detr
git pull
```

## Setup

```bash
cd ~/huggingface-detr
source Docker/ros1-cpu/run.sh
source environment_ros1.sh
source clean_ros1_ws.sh
source build_ros1_all.sh
source environment_ros1.sh
```

## Demo

```bash
cd ~/huggingface-detr
source Docker/ros1-gpu/run.sh
source environment_ros1.sh
rosrun detr_inference download_model.py
roslaunch detr_inference detr_inference_searching.launch
```

# 3. In robotx-2022 repo:

## Clone Repo

```bash
git clone --recursive git@github.com:ARG-NCTU/robotx-2022.git
``` 

```bash
cd ~/robotx-2022/catkin_ws/src/vrx/
git checkout -b robotx-2022 origin/robotx-2022
``` 

## Update repo and submodules

```bash
cd ~/robotx-2022
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

```bash
cd ~/robotx-2022/catkin_ws/src/vrx/
git pull
```

## Setup

```bash
cd ~/robotx-2022
source ipc_run.sh
source scripts/00_setup_all.sh
cd catkin_ws
catkin build
```

## Demo

```bash
cd ~/robotx-2022
source ipc_run.sh
source scripts/00_setup_all.sh
make buoy_navigation
```