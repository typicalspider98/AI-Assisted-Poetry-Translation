#!/bin/bash

# ===== 1. 检查是否已安装 docker =====
if ! command -v docker &> /dev/null
then
    echo "🔧 Docker 未安装，正在安装..."
    sudo apt update
    sudo apt install -y ca-certificates curl gnupg lsb-release

    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "✅ Docker 安装完成"
else
    echo "✅ Docker 已安装，跳过安装步骤"
fi

# ===== 2. 启动 Docker 服务（根据系统类型）=====
INIT_SYSTEM=$(ps -p 1 -o comm=)
echo "🧪 当前 init system: $INIT_SYSTEM"

if [ "$INIT_SYSTEM" = "systemd" ]; then
    echo "🔧 使用 systemctl 启动 Docker 服务..."
    sudo systemctl start docker || echo "⚠️ 启动失败，请检查 Docker 服务状态"
else
    echo "⚠️ 当前系统不是 systemd，尝试启动 dockerd（如已运行将跳过）"
    if ! pgrep dockerd > /dev/null; then
        nohup dockerd > docker.log 2>&1 &
        sleep 3
        echo "✅ dockerd 启动成功（后台运行）"
    else
        echo "✅ dockerd 已在运行，跳过启动"
    fi
fi

# ===== 3. 构建镜像（自动使用缓存）=====
echo "🔧 构建 Docker 镜像..."
docker build -t poetry-translator-gpu .

# ===== 4. 检查 GPU runtime 支持 =====
USE_GPU=false
if docker info | grep -q 'Runtimes:.*nvidia'; then
    USE_GPU=true
fi

# ===== 5. 检查容器是否存在 =====
if docker ps -a --format '{{.Names}}' | grep -q '^redis-server_NZDdictionary$'; then
    echo "⚠️ 容器 redis-server_NZDdictionary 已存在，尝试启动它..."
    docker start redis-server_NZDdictionary
    docker exec -it redis-server_NZDdictionary bash
else
    echo "🚀 创建并启动新容器 redis-server_NZDdictionary..."
    if [ "$USE_GPU" = true ]; then
        echo "✅ 检测到 GPU 支持，使用 --gpus all 启动"
        docker run -it --gpus all -p 6379:6379 --name redis-server_NZDdictionary poetry-translator-gpu
    else
        echo "⚠️ 未检测到 GPU runtime，降级为 CPU 模式启动"
        docker run -it -p 6379:6379 --name redis-server_NZDdictionary poetry-translator-gpu
    fi
fi

# ===== 6. 结束提示 =====
echo "✅ 安装完成！如需再次进入容器，请运行："
echo "docker exec -it redis-server_NZDdictionary bash"
