# Build upon nvidia docker file
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Just in case we need it
ENV DEBIAN_FRONTEND noninteractive

# Install Miniconda for virtual environments
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

# Make conda commands executable in shell:
RUN conda init bash

# Create virtual environment in conda
RUN conda create -n attentive_cylinder_3d python=3.9

# automatically activate environment (only necessary when entering container with bash)
RUN echo "conda activate attentive_cylinder_3d" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "vista", "/bin/bash", "-c"]

# Install apt dependencies
RUN apt update && apt upgrade -y
RUN apt install -y git wget unzip libboost-all-dev cmake build-essential fmpeg libsm6 libxext6

# Install conda dependencies
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
RUN conda install python=3.9.2 numpy tqdm pyyaml numba strictyaml -c conda-forge

# Install pip dependencies
RUN pip3 install --upgrade pip
RUN pip3 install cython==0.29.24
RUN pip3 install nuscenes-devkit==1.1.6
RUN pip3 install spconv-cu114
RUN pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0%2Bcu116.html

# Git config
RUN git config --global user.email "attentive_cylinder_3d_docker@test.net"
RUN git config --global user.name "attentive_cylinder_3d_docker"

# Environment variable for cuda
ENV CUDA_ROOT=/usr/local/cuda

## Install Attentive Cylinder 3D
RUN git clone --recursive --depth 1 https://github.com/nerovalerius/AttentiveCylinder3D.git
WORKDIR /workspace/

# Make net executable
RUN chmod +x train_nusc.sh
RUN chmod +x train.sh

# create folders - they must be created since they are inside .gitignore and not wanted in the online repo
RUN mkdir -p /models/load/
RUN mkdir -p /models/save/
RUN mkdir -p /save_folder/
RUN mkdir -p /dataset/semanticKITTI
RUN mkdir -p /dataset/nuScenes

# Instead of using conda activate, there’s another way to run a command inside an environment. 
# conda run -n myenv yourcommand will run yourcommand inside the environment. 
# You’ll also want to pass the --no-capture-output flag to conda run so it streams stdout and stderr.
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "attentive_cylinder_3d", "python", "-u", "train_cylinder_asym.py"]

