# Base image
ARG PYTORCH="1.13.0"
ARG CUDA="11.6"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV DEBIAN_FRONTEND noninteractive

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Core tools
RUN apt-get update && apt-get install -y \
    cmake \
    curl \
    docker.io \
    ffmpeg \
    git \
    htop \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    ninja-build \
    unzip \
    vim \
    wget \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# Python tools
RUN pip install --upgrade pip
RUN pip install \
    awscli==1.33.40 \
    boto3==1.34.158 \
    coloredlogs==15.0.1 \
    gdown==5.2.0 \
    gpustat==1.1.1 \
    hydra-core==1.3.2 \
    imageio==2.34.2 \
    imageio-ffmpeg==0.5.1 \
    memory_profiler==0.61.0 \
    nevergrad==1.0.3 \
    numpy==1.26.4 \
    omegaconf==2.3.0 \
    openmim==0.3.9 \
    pandas==2.2.2 \
    psutil==5.9.0 \
    pycocotools==2.0.8 \
    seaborn==0.13.2 \
    wandb==0.17.6

# Install OpenMMLab packages
ENV FORCE_CUDA="1"

# Custom nuscenes-devkit
RUN git clone https://github.com/TRAILab/nuscenes-devkit.git && cd nuscenes-devkit && \
    pip install ./setup

# Optional: SimpleTrack for BEV visualization
RUN git clone https://github.com/tusen-ai/SimpleTrack && cd SimpleTrack && \
    pip install -e .

RUN mim install mmengine "mmcv==2.1.0" "mmdet==3.2.0" "mmdet3d==1.4.0"

WORKDIR /workspace/jdt3d
ENV PYTHONPATH="/workspace/jdt3d:${PYTHONPATH}"
RUN git config --global --add safe.directory /workspace/jdt3d
