FROM python:3.10.14-slim

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt update && apt install -y --no-install-recommends git \
    && apt clean autoclean \
    && apt autoremove -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

COPY requirements-tool.txt ./

RUN pip install -r requirements-tool.txt

COPY .pre-commit-config.yaml ./
RUN git init && pre-commit install-hooks
