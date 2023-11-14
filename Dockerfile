# Dockerfile
#
# Created on Tue Nov 14 2023 by Florian Pfleiderer
#
# Copyright (c) 2023 TU Wien

# Use an official PyTorch base image
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set the working directory in the container
WORKDIR /cyws3d

# Copy the current directory contents into the container at /app
COPY . .

# Install git and wget
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y wget

# Install miniconda.
ENV CONDA_DIR $HOME/miniconda3
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
# Make non-activate conda commands available.
ENV PATH=$CONDA_DIR/bin:$PATH

# run updates
RUN conda update -n base -c defaults conda

# Create a Conda environment
# RUN conda create -n cyws3d python=3.9 -y
RUN conda env create -f environment.yml
RUN echo "conda activate cyws3d" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Install dependencies
# RUN conda install -n cyws3d -c pytorch pytorch=1.10.1 torchvision=0.11.2 cudatoolkit=11.3.1
# RUN conda install -n cyws3d -c fvcore -c iopath -c conda-forge fvcore iopath
# RUN conda install -n cyws3d -c pytorch3d pytorch3d=0.7.1 --freeze-installed
# RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
# RUN pip install timm==0.6.12 jsonargparse matplotlib imageio loguru einops wandb easydict kornia==0.6.8 scipy etils mmdet==2.25.3
# RUN pip install segmentation-models-pytorch@git+https://github.com/ragavsachdeva/segmentation_models.pytorch.git@2cde92e776b0a074d5e2f4f6a50c68754f948015

# Set the default command to run when the container starts
CMD ["bash"]
