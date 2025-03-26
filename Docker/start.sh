#!/bin/bash
echo "🔧 构建 Docker 镜像..."
docker build -t poetry-translator-gpu .

echo "🚀 启动容器并挂载 GPU，设置容器名为 redis-server_NZDdictionary"
docker run -it --rm --gpus all -p 6379:6379 --name redis-server_NZDdictionary poetry-translator-gpu

docker exec -it poetry-translator-gpu bash