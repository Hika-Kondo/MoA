FROM ubuntu:20.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install 
RUN apt-get update --fix-missing && \
    apt-get install -y \
    wget \
    git \
    gcc \
    g++ \
    make \
    gosu

ARG UID
ARG GID
ARG UNAME

ENV UID ${UID}
ENV GID ${GID}
ENV UNAME ${UNAME}

RUN groupadd -g ${GID} ${UNAME}
RUN useradd -u ${UID} -g ${UNAME} -m ${UNAME}

# COPY ./setVolumePermission.sh /home/${UNAME}
# RUN chmod +x /home/${UNAME}/setVolumePermission.sh

# ENTRYPOINT ["sh", "-c", "/home/${UNAME}/setVolumePermission.sh"]

ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:${PATH}
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh
RUN conda install -y conda

RUN conda install -c anaconda numpy==1.19.1 && \
    conda install -c conda-forge matplotlib==3.3.2 notebook lightgbm xgboost && \
    conda install pandas

# vim key bind
# Create required directory in case (optional)
RUN mkdir -p $(jupyter --data-dir)/nbextensions && \
    cd $(jupyter --data-dir)/nbextensions && \
    git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding 

WORKDIR /kaggle
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8000", "--NotebookApp.token=''", "--NotebookApp.password=''", "--allow-root"]
