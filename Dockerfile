# Build upon nvidia docker file
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Just in case we need it
ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install apt dependencies
RUN apt update && apt upgrade -y
RUN apt install -y git wget unzip libboost-all-dev cmake build-essential ffmpeg libsm6 libxext6 ninja-build

# Install Miniconda for virtual environments
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm Miniconda3-latest-Linux-x86_64.sh

# In order to use the conda command
ENV PATH=/root/miniconda3/bin:${PATH}

# Make conda commands executable in shell:
RUN conda init bash

# Create virtual environment in conda
RUN conda create -n attentivecylinder3d python=3.9

# For install of Minkowski Engine with CUDA
RUN ln -s /usr/bin/python3.9 /usr/bin/python && ln -s /usr/bin/pip3 /usr/bin/pip

# automatically activate environment (only necessary when entering container with bash)
RUN echo "conda activate attentivecylinder3d" >> ~/.bashrc

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "attentivecylinder3d", "/bin/bash", "-c"]

# Install conda dependencies
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda install python=3.9.2 numpy tqdm pyyaml numba strictyaml -c conda-forge
RUN conda install openblas-devel -c anaconda
RUN conda install jupyterlab -c conda-forge

# Install pip dependencies
RUN pip3 install --upgrade pip
RUN pip3 install cython nuscenes-devkit spconv-cu117
RUN pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html
RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html
#RUN pip3 install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda" --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# Environment variable for cuda
ENV CUDA_ROOT=/usr/local/cuda

## Install Attentive Cylinder 3D
WORKDIR /workspace/


