FROM ubuntu:20.04

SHELL [ "/bin/bash", "--login", "-c" ]

RUN apt-get update \
  && apt-get install -y wget

WORKDIR /cyws3d

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

# Create and activate the environment.
COPY environment.yml .
RUN conda env create -f environment.yml

# copy folder structure
COPY utils.py config.yml inference.py ./
COPY demo_data/ ./demo_data/
COPY modules/ ./modules/
COPY SuperGluePretrainedNetwork/ ./SuperGluePretrainedNetwork/
