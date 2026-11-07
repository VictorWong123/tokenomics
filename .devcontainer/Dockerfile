FROM mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv python3-pip python3-dev pkg-config curl ca-certificates time \
    && rm -rf /var/lib/apt/lists/*
USER vscode
