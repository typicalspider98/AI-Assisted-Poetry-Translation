#!/bin/bash

# ====== 1. 安装 Docker ======
if ! command -v docker &> /dev/null
then
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

# ====== 2. 启动 Docker（根据系统类型）======
INIT_SYSTEM=$(ps -p 1 -o comm=)
echo "🧪 当前 init system: $INIT_SYSTEM"

if [ "$INIT_SYSTEM" = "systemd" ]; then
    echo "🔧 尝试通过 systemctl 启动 Docker 服务..."
    sudo systemctl start docker || echo "⚠️ systemctl 启动失败，请手动检查 Docker 服务是否已运行"
else
    echo "⚠️ 当前系统未使用 systemd（而是 $INIT_SYSTEM）"
    echo "🔄 尝试通过 dockerd 启动 Docker..."
    nohup dockerd > docker.log 2>&1 &
    sleep 3
    echo "✅ dockerd 已在后台启动（日志写入 docker.log）"
fi


# ====== 3. 安装 NVIDIA Container Toolkit（如果未安装）======
echo "🔍 检查是否已安装 nvidia-container-toolkit..."
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "🔧 未安装，开始安装 NVIDIA Container Toolkit..."

    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    sudo apt update
    sudo apt install -y nvidia-container-toolkit

    echo "✅ NVIDIA Container Toolkit 安装完成"
else
    echo "✅ NVIDIA Container Toolkit 已安装，跳过"
fi

# ====== 4. 配置 /etc/docker/daemon.json（仅当未配置时）======
DOCKER_DAEMON_FILE="/etc/docker/daemon.json"

if [ ! -f "$DOCKER_DAEMON_FILE" ] || ! grep -q '"nvidia"' "$DOCKER_DAEMON_FILE"; then
    echo "⚙️ 正在写入 GPU runtime 到 $DOCKER_DAEMON_FILE..."
    sudo bash -c "cat > $DOCKER_DAEMON_FILE" <<EOF
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

# ====== 5. 重启 Docker 服务（如果支持）======
if [ "$INIT_SYSTEM" = "systemd" ]; then
    echo "🔁 重启 Docker 服务以生效 GPU 支持..."
    sudo systemctl restart docker
    echo "✅ Docker 已重启"
else
    echo "⚠️ 非 systemd 系统，请手动重启 Docker（或确保 dockerd 正常）"
fi

echo "🎉 Docker + GPU 环境准备完成！"

# chmod +x step1-docker-install.sh
# ./step1-docker-install.sh
