# Dockerfile
#
# Created on Tue Nov 14 2023 by Florian Pfleiderer
#
# Copyright (c) 2023 TU Wien

# Use an official PyTorch base image
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# use ubuntu base image because work is done in conda environment anyway
FROM ubuntu:20.04

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
# Make conda activate command available from /bin/bash --login shells.
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh" >> ~/.profile
# Make conda activate command available from /bin/bash --interative shells.
RUN conda init bash

# start shell in login mode
SHELL ["/bin/bash", "--login", "-c"]

# run updates
RUN conda update -n base -c defaults conda

# Create a Conda environment
RUN conda env create -f environment.yml

# Install dependencies
RUN conda activate cyws3d && \
    pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html

# start container in cyws3d env
RUN touch ~/.bashrc && echo "conda activate cyws3d" >> ~/.bashrc

# Set the default command to run when the container starts
CMD ["bash"]
