#!/bin/bash

# 检查是否已安装 docker
if ! command -v docker &> /dev/null
then
    echo "🔧 Docker 未安装，正在安装..."
    sudo apt update
    sudo apt install -y ca-certificates curl gnupg lsb-release

    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) \
      signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "✅ Docker 安装完成"
else
    echo "✅ Docker 已安装，跳过安装步骤"
fi

# 启动 docker 服务（可选）
sudo systemctl start docker

# 构建并运行容器
echo "🔧 构建 Docker 镜像..."
docker build -t poetry-translator-gpu .

echo "🚀 启动容器并挂载 GPU，设置容器名为 redis-server_NZDdictionary"
docker run -it --rm --gpus all -p 6379:6379 --name redis-server_NZDdictionary poetry-translator-gpu

# 提示用户如何进入容器
echo "✅ 安装完成！如需再次进入容器，请运行："
echo "docker exec -it poetry-translator-gpu bash"

# Step:
# chmod +x start.sh
# ./start.sh