#!/bin/bash

# 检查是否已安装 docker
if ! command -v docker &> /dev/null
then
    echo "🔧 Docker 未安装，正在安装..."
    apt update
    apt install -y ca-certificates curl gnupg lsb-release

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
      gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    if [ ! -f /etc/apt/sources.list.d/docker.list ]; then
        echo "➕ 添加 Docker APT 源..."
        echo \
          "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
          https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
          tee /etc/apt/sources.list.d/docker.list > /dev/null
    else
        echo "✅ Docker APT 源已存在，跳过添加"
    fi

    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "✅ Docker 安装完成"
else
    echo "✅ Docker 已安装，跳过安装步骤"
fi

# 检查当前是否使用 systemd
INIT_SYSTEM=$(ps -p 1 -o comm=)
echo "🧪 当前 init system: $INIT_SYSTEM"

if [ "$INIT_SYSTEM" = "systemd" ]; then
    echo "🔧 尝试通过 systemctl 启动 Docker 服务..."
    sudo systemctl start docker || echo "⚠️ systemctl 启动失败，请手动检查 Docker 服务是否已运行"
else
    echo "⚠️ 当前系统未使用 systemd（而是 $INIT_SYSTEM）"
    if pgrep dockerd > /dev/null; then
        echo "✅ dockerd 已在运行，跳过启动"
    else
        echo "🔄 尝试通过 dockerd 启动 Docker..."
        nohup dockerd > docker.log 2>&1 &
        sleep 3
        echo "✅ dockerd 已在后台启动（日志写入 docker.log）"
    fi
fi

# chmod +x step1-docker-install.sh
# ./step1-docker-install.sh