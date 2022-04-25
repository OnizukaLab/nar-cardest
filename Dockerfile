FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

WORKDIR /workspaces/nar-cardest
RUN apt update \
 && apt install -y --no-install-recommends \
      tmux \
      htop \
      git \
      curl \
      ca-certificates \
      openssh-client \
      python3-pip \
      python3.8 \
      python3.8-distutils \
      libpython3.8-dev \
 && apt clean \
 && rm -rf /var/lib/apt/lists/*

RUN ln -f -s $(which python3.8) $(dirname $(which python3.8))/python3
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3 -
ENV PATH=$PATH:/root/.poetry/bin PYTHONPATH=/workspaces/nar-cardest
COPY pyproject.toml poetry.toml poetry.lock ./
RUN poetry install

CMD poetry shell
