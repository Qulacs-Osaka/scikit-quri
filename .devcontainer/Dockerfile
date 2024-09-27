FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11-bullseye

ENV DEBIAN_FRONTEND=noninteractive

# Requires $POETRY_HOME to install the location; poetry is installed under /root by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        libboost-dev \
    && curl -sSL https://install.python-poetry.org | POETRY_HOME=/home/vscode/.local python3 -
