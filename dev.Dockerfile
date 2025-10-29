FROM python:3.11-slim as test-image
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Minimal OS deps for tests
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create venv and activate permanently
RUN uv venv --clear --python 3.13
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the project with dev extras
COPY pyproject.toml README.md ./
COPY qdax ./qdax
RUN uv pip install jax[cuda] .[dev]


FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04 as cuda-image
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# System packages needed for examples/headless rendering and tooling
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
    screen \
    sudo \
    tmux \
    unzip \
    vim \
    wget \
    nano \
    xvfb && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create venv and activate permanently
RUN uv venv --clear --python 3.13
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install the project with dev and examples extras
COPY pyproject.toml README.md ./
COPY qdax ./qdax
RUN uv pip install jax[cuda] .[dev,examples]


FROM cuda-image as dev-image
# The dev-image already contains dependencies; repo can be mounted at /app for live dev

ARG USER_ID=1000
ARG GROUP_ID=1000
ENV USER=eng
ENV GROUP=eng
RUN groupadd --gid ${GROUP_ID} $GROUP && useradd -g $GROUP --uid ${USER_ID} --shell /usr/sbin/nologin -m $USER  && chown -R $USER:$GROUP /app
USER $USER


FROM cuda-image as run-image
# Default run image
WORKDIR /
CMD ["python"]
