FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04

ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ffmpeg \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libmpich-dev \
    libosmesa6-dev \
    mpich \
    patchelf \
    python3-opengl \
    python3-dev=3.8* \
    python3-pip \
    screen \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

# Install (mini)conda. This will enable us to install python packages
# for all users without overriding or impacting in any way the python packages installed by
# root. This is important because the user we will end up using is parametrized by a dockerfile
# argument (see USER_ID later in this file) and we want to share the docker image cache as much
# as possible for all possible values of USER_ID.
RUN curl https://repo.anaconda.com/pkgs/misc/gpgkeys/anaconda.asc | gpg --dearmor > conda.gpg
RUN install -o root -g root -m 644 conda.gpg /usr/share/keyrings/conda-archive-keyring.gpg
RUN gpg --keyring /usr/share/keyrings/conda-archive-keyring.gpg --no-default-keyring --fingerprint 34161F5BF5EB1D4BFBBB8F0A8AEB4F8B29D82806
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/conda-archive-keyring.gpg] https://repo.anaconda.com/pkgs/misc/debrepo/conda stable main" > /etc/apt/sources.list.d/conda.list

RUN apt-get update && \
    apt-get install -y --no-install-recommends conda=4.9.2-0 && \
    rm -rf /var/lib/apt/lists/*

# Install all pip dependencies through conda's global pip.
# Also manually list here all mujoco_py pip dependencies before installing
# mujoco_py itself as doing otherwise leads to errors (only observed for mujoco_py < 2.*)
# Note that we pin the version of acme to a specific commit as an imcompatibility with launchpad's
# released versions available through pip has been introduced after, specifically as part
# of commit 48e6991c922ac59873c7c74e44b918683244e9ef
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.4/targets/x86_64-linux/lib
ENV PATH=/opt/conda/bin:$PATH

COPY ./requirements.txt /tmp/requirements.txt
COPY ./requirements-dev.txt /tmp/requirements-dev.txt

RUN pip3 --no-cache-dir install -r /tmp/requirements.txt \
    jaxlib==0.1.75+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_releases.html \
    torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html \
    -r /tmp/requirements-dev.txt \
    && rm -rf /tmp/*


# Create 'eng' user (unfortunately mujoco 1.5 needs to be installed by a user - not root)
# The id and group-id of 'eng' can be parametrized to match that of the user that will use this
# docker image so that the eng can create files in mounted directories seamlessly (without
# permission issues)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid ${GROUP_ID} eng
RUN useradd -l --gid eng --uid ${USER_ID} --shell /bin/bash --home-dir /app --create-home eng
WORKDIR /app
USER eng


RUN echo "alias ls='ls --color=auto'" >> .bashrc
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> .bashrc

ENV PATH=/app/bin:/app/.local/bin:$PATH
# Disable debug, info, warning, and error tensorflow logs
ENV TF_CPP_MIN_LOG_LEVEL=3
#ENV JAX_PLATFORM_NAME="cpu"

RUN mkdir qdax

# Add symlink to qdax python package so that users do not have to
# run scripts with the python -m option.
USER root
RUN ln -s /app/qdax/qdax /opt/conda/lib/python3.8/site-packages/qdax
USER eng
WORKDIR /app/qdax
