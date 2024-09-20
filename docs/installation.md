# Installation

## Setting up the environment
There are three options for installing the dependencies using either `docker`, `singularity` or `conda` depending on your preference.


### Using docker

If it is not already done, install docker. Follow [this tutorial](https://docs.docker.com/engine/install/ubuntu/) to install it on Ubuntu.

Steps to build your docker image:

- Build the dev-image in `dev.Dockerfile`

```
sudo docker build --target dev-image -t instadeep/qdax:$USER -f dev.dockerfile .
```

!!! hint "Save time: use the Makefile"
    The command `make build` automatically builds the container.

- Setup your image, mount your directory and start developing

To develop code in your IDE while running it inside your container, mount this folder into your container. For example, you might first get the path to your QDax root folder inside an environment variable:

```
cd ..
export QDAX_PATH=$(pwd)
```

and then mount QDax into your container:

```
cd docker
sudo docker run --rm -it -v $QDAX_PATH:/app instadeep/qdax:$USER /bin/bash
```

!!! hint "Save time: use the Makefile"
    The command `make dev_container` automatically runs the image, mounts the qdax directory, setups the GPU and forwards Neptune's credentials. Pretty handy, right?


??? question "How to use GPUs with Docker?"

    Docker also allows you to use GPUs. First, be sure you have already installed
    the Nvidia drivers on your computer and that you have installed
    [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
    Then, to use all your GPUs inside docker run:

    ```
    sudo docker run --rm -it --gpus all -v $QDAX_PATH:/app instadeep/qdax:$USER /bin/bash
    ```

    If you want to use only a subset of the available GPUs inside docker (for instance only GPUs with ID 0 and 1), run:

    ```
    sudo docker run --rm -it --gpus '"device=0,1"' -v $QDAX_PATH:/app instadeep/qdax:$USER /bin/bash
    ```

### Using conda

1. If it is not already done, install conda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

2. Install some necessary packages on your machine
```zsh
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
                    ffmpeg \
                    libgl1-mesa-dev \
                    libgl1-mesa-glx \
                    libglfw3 \
                    libosmesa6-dev \
                    patchelf \

```

1. Clone `qdax`
```zsh
git clone git@github.com:adaptive-intelligent-robotics/QDax.git
```

1. Create a conda environment with all required libraries
    ```zsh
    cd QDax
    conda env create -f environment.yaml
    ```

2. Activate the environment and manually install the package qdax
    ```zsh
    conda activate qdaxpy310
    pip install -e .
    ```

!!! warning "Install GPU support"
    This tutorial only covers the installation on a CPU hardware. If you wish to run the
    code on GPU, consider using the docker tutorial. You can also install cuda on your
    machine and follow the instructions to setup jax on GPU from
    [here](https://github.com/google/jax#installation).
