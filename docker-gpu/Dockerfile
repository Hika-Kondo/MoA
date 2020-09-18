FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install 
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    vim \
    mercurial \
    subversion \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    gcc \
    g++

# Add OpenCL ICD files for LightGBM
RUN mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

RUN apt-get install -y cmake 3.10.2

ARG UID
ARG GID
ARG UNAME

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}

RUN groupadd -g ${GID} ${UNAME}
RUN useradd -u ${UID} -g ${UNAME} -m ${UNAME}

RUN cd /usr/local/src && mkdir lightgbm && cd lightgbm && \
    git clone --recursive --branch stable --depth 1 https://github.com/microsoft/LightGBM && \
    cd LightGBM && mkdir build && cd build && \
    cmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ .. && \
    make OPENCL_HEADERS=/usr/local/cuda-8.0/targets/x86_64-linux/include LIBOPENCL=/usr/local/cuda-8.0/targets/x86_64-linux/lib

ENV PATH /usr/local/src/lightgbm/LightGBM:${PATH}

RUN cd /usr/local/src/lightgbm/LightGBM/python-package && \
    python setup.py install --precompile

# vim key bind
# Create required directory in case (optional)
RUN mkdir -p $(jupyter --data-dir)/nbextensions && \
    cd $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding 

RUN conda install -c anaconda numpy==1.19.1 scipy && \
    conda install -c conda-forge matplotlib==3.3.2 notebook && \
    conda install pandas

WORKDIR /kaggle
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8000", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
