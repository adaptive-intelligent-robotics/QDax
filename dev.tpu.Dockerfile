FROM mambaorg/micromamba:0.22.0 as conda

# Speed up the build, and avoid unnecessary writes to disk
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements_tpu.txt /tmp/requirements_tpu.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY environment_tpu.yaml /tmp/environment_tpu.yaml


RUN micromamba create -y --file /tmp/environment_tpu.yaml \
    && micromamba clean --all --yes \
    && find /opt/conda/ -follow -type f -name '*.pyc' -delete


FROM debian:bullseye-slim as test-image

COPY --from=conda /opt/conda/envs/. /opt/conda/envs/
ENV PATH=/opt/conda/envs/squidpy38/bin/:$PATH APP_FOLDER=/app
ENV PYTHONPATH=$APP_FOLDER:$PYTHONPATH
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR $APP_FOLDER

FROM test-image as dev-image
# The run-image (default) is the same as the dev-image with the some files directly
# copied inside
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
    python3-dev \
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
