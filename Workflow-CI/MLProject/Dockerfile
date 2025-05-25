FROM python:3.12

# Set working directory di dalam container
WORKDIR /app

# Copy conda environment file dan requirements.txt
COPY conda.yaml .
COPY requirements.txt . 

# Install miniconda dan buat environment dari conda.yaml
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/conda/bin:${PATH}"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN conda env create -f conda.yaml && conda clean --all

# Aktifkan environment dan install pip dependencies
ENV CONDA_DEFAULT_ENV=pollution_env
ENV PATH /opt/conda/envs/pollution_env/bin:$PATH

# Copy semua file project MLflow Anda
COPY . .

