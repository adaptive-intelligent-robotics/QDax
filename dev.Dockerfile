FROM mambaorg/micromamba:1.5.1 as conda

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 CONDA_DIR=/opt/conda
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV MAMBA_ROOT_PREFIX="/opt/conda"

COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY environment.yaml /tmp/environment.yaml

RUN micromamba create -y --file /tmp/environment.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete


FROM python as test-image
ENV PATH=/opt/conda/envs/qdaxpy310/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
COPY requirements-dev.txt ./

RUN pip install -r requirements-dev.txt


FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04 as cuda-image
ENV PATH=/opt/conda/envs/qdaxpy310/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH


ENV DISTRO ubuntu2004
ENV CPU_ARCH x86_64
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/$DISTRO/$CPU_ARCH/3bf863cc.pub


COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN pip --no-cache-dir install jaxlib==0.4.16+cuda11.cudnn86 \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && rm -rf /tmp/*

WORKDIR $APP_FOLDER

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
RUN groupadd --gid ${GROUP_ID} $GROUP && useradd -g $GROUP --uid ${USER_ID} --shell /usr/sbin/nologin -m $USER  && chown -R $USER:$GROUP $APP_FOLDER
USER $USER


FROM cuda-image as dev-image
# The dev-image does not contain the any file, qdax is expected to be mounted
# afterwards

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libosmesa6-dev \
    patchelf \
    python3-opengl \
    python3-dev=3.10* \
    python3-pip \
    screen \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    nano \
    xvfb && \
    rm -rf /var/lib/apt/lists/*


COPY requirements-dev.txt /tmp/requirements-dev.txt
RUN pip --no-cache-dir install -r /tmp/requirements-dev.txt && rm -rf /tmp/*
USER $USER

FROM cuda-image as run-image
# The run-image (default) is the same as the dev-image with the some files directly
# copied inside

COPY qdax qdax
COPY setup.py ./
COPY README.md ./

RUN pip install .

WORKDIR /

CMD ["python"]
