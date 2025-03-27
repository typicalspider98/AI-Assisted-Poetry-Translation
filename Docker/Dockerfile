FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git redis-server redis-tools wget curl bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    && apt-get clean

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && rm miniconda.sh

WORKDIR /workspace
RUN git clone https://github.com/typicalspider98/AI-Assisted-Poetry-Translation.git /workspace

COPY environment.yml /tmp/environment.yml
RUN conda update -n base -c defaults conda && \
    conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "deepseek-r1", "/bin/bash", "-c"]

# 添加 Redis 测试脚本
RUN echo -e '#!/bin/bash\nredis-cli ping' > /check_redis.sh && chmod +x /check_redis.sh

#CMD service redis-server start && bash
CMD service redis-server start && tail -f /dev/null
