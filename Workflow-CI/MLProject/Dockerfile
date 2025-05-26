FROM python:3.12

# Set working directory di dalam container
WORKDIR /app

# Copy file penting
COPY conda.yaml .
COPY requirements.txt .
COPY model.pkl .
COPY inference.py .
COPY pollution_dataset_preprocessed_advance.csv .

# Install dependensi sistem
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV PATH="/opt/conda/bin:${PATH}"
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Buat dan aktifkan environment conda
RUN conda env create -f conda.yaml && conda clean --all
ENV CONDA_DEFAULT_ENV=pollution_env
ENV PATH /opt/conda/envs/pollution_env/bin:$PATH

# Install dependensi tambahan untuk Flask dan Prometheus
RUN pip install flask prometheus_client psutil pandas joblib scikit-learn

# Jalankan inference.py
CMD ["python", "inference.py"]