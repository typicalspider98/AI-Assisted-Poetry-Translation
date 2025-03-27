#!/bin/bash

# ====== 1. 安装 Docker ======
if ! command -v docker &> /dev/null; then
    echo "🔧 Docker 未安装，正在安装..."
    apt update
    apt install -y ca-certificates curl gnupg lsb-release

    mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
      tee /etc/apt/sources.list.d/docker.list > /dev/null

    apt update
    apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "✅ Docker 安装完成"
else
    echo "✅ Docker 已安装，跳过安装步骤"
fi

# ====== 2. 安装 NVIDIA Container Toolkit（如果未安装）======
echo "🔍 检查是否已安装 nvidia-container-toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "🔧 未安装，开始安装 NVIDIA Container Toolkit..."

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
        tee /etc/apt/sources.list.d/nvidia-docker.list

    apt update
    apt install -y nvidia-container-toolkit

    echo "✅ NVIDIA Container Toolkit 安装完成"
else
    echo "✅ NVIDIA Container Toolkit 已安装，跳过"
fi

# ====== 3. 配置 /etc/docker/daemon.json（仅当未配置）======
DOCKER_DAEMON_FILE="/etc/docker/daemon.json"
if [ ! -f "$DOCKER_DAEMON_FILE" ] || ! grep -q '"nvidia"' "$DOCKER_DAEMON_FILE"; then
    echo "⚙️ 正在写入 GPU runtime 到 $DOCKER_DAEMON_FILE..."
    cat > "$DOCKER_DAEMON_FILE" <<EOF
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF
    echo "✅ daemon.json 配置完成"
else
    echo "✅ daemon.json 已包含 nvidia runtime，跳过修改"
fi

# ====== 4. 启动 Docker（根据系统类型）======
INIT_SYSTEM=$(ps -p 1 -o comm=)
echo "🧪 当前 init system: $INIT_SYSTEM"

if [ "$INIT_SYSTEM" = "systemd" ]; then
    echo "🔁 使用 systemctl 重启 Docker 服务..."
    systemctl restart docker && echo "✅ Docker 已重启"
else
    echo "⚠️ 当前系统未使用 systemd（而是 $INIT_SYSTEM）"

    echo "🔄 终止已有 dockerd（如有）..."
    pkill dockerd 2>/dev/null
    echo "🔄 启动新的 dockerd（后台运行）..."
    nohup dockerd > docker.log 2>&1 &
    sleep 3
    echo "✅ dockerd 已在后台启动（日志写入 docker.log）"
fi

# ====== 5. 检查 GPU Runtime 是否生效 ======
echo "🔍 当前 Docker Runtime 支持情况："
docker info | grep -i runtime

echo "🎉 Docker + NVIDIA GPU 环境准备完成！你现在可以使用 --gpus all 启动容器。"

# 使用方式：
# chmod +x step1-docker-install.sh
# ./step1-docker-install.sh
