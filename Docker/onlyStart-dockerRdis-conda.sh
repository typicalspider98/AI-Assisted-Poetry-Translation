#!/bin/bash

# ===== 项目配置 =====
CONDA_ENV_NAME="deepseek-r1"
ENV_YML="environment.yml"
CONTAINER_NAME="redis-server_NZDdictionary"
GIT_REPO_URL="https://github.com/typicalspider98/AI-Assisted-Poetry-Translation.git"
PROJECT_DIR="AI-Assisted-Poetry-Translation"

# ===== 1. 安装 Docker（如未安装） =====
if ! command -v docker &> /dev/null; then
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

# ===== 2. 启动 Docker（兼容非 systemd）=====
INIT_SYSTEM=$(ps -p 1 -o comm=)
if [ "$INIT_SYSTEM" = "systemd" ]; then
    sudo systemctl start docker
else
    if ! pgrep dockerd > /dev/null; then
        nohup dockerd > docker.log 2>&1 &
        sleep 3
    fi
fi

# ===== 3. 启动 Redis 容器 =====
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "⚠️ Redis 容器已存在，尝试启动..."
    docker start $CONTAINER_NAME
else
    echo "🚀 创建并启动 Redis 容器..."
    docker run -d --name $CONTAINER_NAME -p 6379:6379 redis:7
fi

# ===== 4. 创建 Conda 环境（如未存在）=====
if conda info --envs | grep -q "^$CONDA_ENV_NAME"; then
    echo "✅ Conda 环境 '$CONDA_ENV_NAME' 已存在"
else
    echo "📦 正在创建 Conda 环境 '$CONDA_ENV_NAME'..."
    conda env create -f $ENV_YML -n $CONDA_ENV_NAME
fi

# ===== 5. 安装 Git 和 Git LFS（如未安装）=====
if ! command -v git &> /dev/null; then
    echo "📥 安装 Git..."
    sudo apt install -y git
fi

if ! command -v git-lfs &> /dev/null; then
    echo "📥 安装 Git LFS..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt install -y git-lfs
    git lfs install
else
    echo "✅ Git LFS 已安装"
fi

# ===== 6. Clone 项目代码（保持幂等）=====
if [ -d "$PROJECT_DIR" ]; then
    echo "✅ 项目目录 '$PROJECT_DIR' 已存在，跳过 clone"
else
    echo "📥 Clone 项目代码中..."
    git clone "$GIT_REPO_URL"
fi

# ===== 7. 提示完成 =====
echo "🎉 所有步骤完成！你现在可以激活环境运行程序："
echo "conda activate $CONDA_ENV_NAME"
echo "cd $PROJECT_DIR && python your_script.py"
echo "或者使用 redis-cli 测试 Redis：redis-cli -h localhost -p 6379 ping"

# chmod +x only-dockerRdis-conda-git.sh
# ./only-dockerRdis-conda-git.sh