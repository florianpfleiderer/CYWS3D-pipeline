# Dockerfile
# Created on Wed Nov 15 2023 by Florian Pfleiderer
# Copyright (c) MIT License

# use ubuntu base image because work is done in conda environment anyway
# FROM ubuntu:20.04

# nvidia cuda base image
FROM nvidia/cuda:11.6.1-base-ubuntu20.04

# Set the working directory in the container
WORKDIR /cyws3d-pipeline

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install git and wget
RUN apt-get update && \
    #cyws3d dependencies
    apt-get install -y --no-install-recommends --no-install-suggests \
    git wget openssh-client \
    # cyws3d dependencies
    libgl1-mesa-glx libglib2.0-0 \ 
    # pipelines dependencies
    libgomp1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Reset noninteractive mode
ENV DEBIAN_FRONTEND=dialog

# Install miniconda.
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# Make non-activate conda commands available.
ENV PATH=$CONDA_DIR/bin:$PATH
# Make conda activate command available from /bin/bash --login shells.
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# Make conda activate command available from /bin/bash --interative shells.
RUN conda init bash

# start shell in login mode
SHELL ["/bin/bash", "--login", "-c"]

# run updates
RUN conda update -n base -c defaults conda

# # Create a Conda environment
RUN conda create -n cyws3d-pipeline python=3.9 -y


## Install conda dependencies
# cyws3d
RUN conda install -n cyws3d-pipeline -c pytorch pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3.1 -y
RUN conda install -n cyws3d-pipeline -c fvcore -c iopath -c conda-forge fvcore iopath -y
RUN conda install -n cyws3d-pipeline pytorch3d==0.7.1 -c pytorch3d --freeze-installed -y
# pipeline


## install pip dependencies
# cyws3d
RUN conda activate cyws3d-pipeline && pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
RUN conda activate cyws3d-pipeline && pip install timm==0.6.12 jsonargparse matplotlib imageio loguru einops wandb easydict kornia==0.6.8 scipy etils mmdet==2.25.3 shapely==2.0.2
RUN conda activate cyws3d-pipeline && pip install segmentation-models-pytorch@git+https://github.com/ragavsachdeva/segmentation_models.pytorch.git@2cde92e776b0a074d5e2f4f6a50c68754f948015
# pipeline
RUN conda activate cyws3d-pipeline && pip install open3d==0.16.0 torchmetrics==1.3.2


# start container in cyws3d-pipeline environment
RUN touch ~/.bashrc && echo "conda activate cyws3d-pipeline" >> ~/.bashrc

# Set the default command to run when the container starts
CMD ["bash"]
