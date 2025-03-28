```shell
# 构建镜像
docker build -t poetry-translator-gpu .

# 运行容器
docker run -it --rm -p 6379:6379 poetry-translator
# 后台运行容器：
docker run -d --name poetry-translator -p 6379:6379 poetry-translator

# 进入：
docker exec -it poetry-translator bash

```