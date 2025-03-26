#!/bin/bash

# 📦 脚本说明：
# 本脚本用于：
# 1. 启动 Redis Docker 容器（如不存在则创建）
# 2. 自动检测并导入 Conda 环境（基于 environment.yml）
# 3. 提示用户进入 Conda 环境运行 Python 项目

# ✅ 检查是否安装 Docker
if ! command -v docker &> /dev/null; then
    echo "❌ 未检测到 Docker，请先安装 Docker 后再运行本脚本"
    exit 1
fi

# ✅ 启动 Redis 容器（如果已存在则跳过）
CONTAINER_NAME=redis-server_NZDdictionary
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "✅ Redis 容器已存在，启动中..."
    docker start $CONTAINER_NAME
else
    echo "🚀 创建并启动 Redis 容器..."
    docker run -d --name $CONTAINER_NAME -p 6379:6379 redis
fi

# ✅ 测试 Redis 是否可用
sleep 2
echo "🔍 检查 Redis 状态..."
docker exec -it $CONTAINER_NAME redis-cli ping

# ✅ 检查是否安装 conda
if ! command -v conda &> /dev/null; then
    echo "❌ 未检测到 Conda，请先安装 Miniconda 或 Anaconda 后再运行本脚本"
    exit 1
fi

# ✅ Conda 环境导入
ENV_NAME="deepseek-r1"
if conda env list | grep -q "^$ENV_NAME"; then
    echo "✅ Conda 环境 '$ENV_NAME' 已存在，跳过导入"
else
    echo "📦 Conda 环境不存在，正在根据 environment.yml 创建..."
    conda env create -f environment.yml
fi

# ✅ 提示激活 Conda 环境
echo "🎯 Conda 环境已准备就绪！"
echo "请运行以下命令开始工作："
echo "  conda activate $ENV_NAME"
echo "  python your_script.py  # 或进入项目目录继续开发"

# chmod +x run_redis_and_conda.sh
# ./run_redis_and_conda.sh

