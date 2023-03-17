FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN DEBIAN_FRONTEND=noninteractive  apt-get update && apt-get install -y --no-install-recommends \
            build-essential  \
            python3          \
            python3-pip      \  
            python3-opencv

RUN pip3 install --no-cache-dir \
    pip                         \ 
    setuptools

RUN python3 -m pip install --upgrade pip

RUN pip3 --no-cache-dir install --upgrade \
    numpy                                 \
    scikit-learn                          \
    torch torchvision                     \
    datasets transformers evaluate        \
    tensorboard                           \
    pillow                                \
    alive-progress                        \
    split-folders

RUN mkdir        /tmp/MPLCONFIGDIR
ENV TORCH_HOME   /tmp
ENV MPLCONFIGDIR /tmp/MPLCONFIGDIR