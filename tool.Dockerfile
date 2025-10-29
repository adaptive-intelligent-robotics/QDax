FROM python:3.11-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt update && apt install -y --no-install-recommends git curl \
    && apt clean autoclean \
    && apt autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create venv and activate permanently
RUN uv venv --clear --python 3.13
ENV VIRTUAL_ENV="/app/.venv"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install just the tooling needed for style checks via uv
RUN uv pip install pre-commit

# Pre-install hooks to speed up CI runs
COPY .pre-commit-config.yaml ./
RUN git init && pre-commit install-hooks
