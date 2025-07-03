FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV ROS_DISTRO=noetic

###################################### user #####################################

ENV SHELL=/bin/bash \
    USER=arg \
    UID=1000 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV HOME=/home/${USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${UID} \
    ${USER} 

RUN echo "root:root" | chpasswd
RUN echo "${USER}:111111" | chpasswd

###################################### basic tools #####################################

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
    && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \ 
    locales \
    cmake \
    unzip \
    make \
    git \
    vim \
    gedit \
    wget \
    sudo \
    lsb-release \
    build-essential \
    net-tools \
    dirmngr \
    gnupg2 \
    mercurial \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libpcap-dev \
    python3-pip \
    python3-setuptools \
    python3-opencv \
    python3-numpy \
    python3-empy  \
    python3-dev \ 
    python3-tk \ 
    apt-transport-https \
    apt-utils \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###################################### ROS #####################################
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros1-latest.list \
    && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-$ROS_DISTRO-desktop-full=1.5.0-1* \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -o Acquire::ForceIPv4=true install --no-install-recommends -y \
    ros-$ROS_DISTRO-desktop-full \
    ros-$ROS_DISTRO-rosbridge-suite \
    ros-$ROS_DISTRO-cv-bridge \
    ros-$ROS_DISTRO-rosbag \
    ros-$ROS_DISTRO-rviz-imu-plugin \
    ros-$ROS_DISTRO-realsense2-camera \
    ros-$ROS_DISTRO-rgbd-launch \
    python3-catkin-tools \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-vcstool \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

###################################### ROSBRIDGE #####################################
RUN python3 -m pip install --upgrade pip \
    roslibpy \
    pymavlink

RUN sudo -H pip install --upgrade twisted \
    cryptography \
    pyOpenSSL

##################################### PIP ######################################

RUN pip3 install \
    matplotlib \
    pandas \
    quart \
    pytest-skip-slow \
    rich \
    opencv-python \
    scikit-learn \
    torch \
    torchvision \
    torchaudio
    
##################################### HuggingFace #####################################

RUN pip3 install \
    datasets \
    evaluate \
    transformers[sentencepiece] \
    accelerate \
    timm \
    albumentations \
    pycocotools

##################################### tensorboard #####################################

RUN pip3 install \
    tensorboard \
    protobuf==3.20.3

##################################### Kalman Filter #####################################

RUN pip3 install \
    filterpy

##################################### setting ###################################################

RUN chown -R ${USER}:${USER} ${HOME}/
RUN echo "${USER} ALL=(ALL)  ALL" >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

USER ${USER}
WORKDIR ${HOME}/huggingface-detr

CMD ["/bin/bash"]