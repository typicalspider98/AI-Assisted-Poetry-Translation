启动一个 Redis 容器，并在本地运行 Redis 服务器。
```
docker run -d --name redis-server_NZDdictionary -p 6379:6379 redis
```
然后检查 Redis 是否运行：
```shell
docker ps
```
> 如果 Redis 在运行，你会看到类似下面的输出
> ```shell
> CONTAINER ID   IMAGE        COMMAND     CREATED         STATUS         PORTS                    NAMES
> cb5710d6eced   redis:latest "redis..."  10 minutes ago Up 10 minutes  0.0.0.0:6379->6379/tcp   redis-server_NZDdictionary
> ```

可以直接 在 Docker 容器内部运行 redis-cli：
```shell
docker exec -it redis-server_NZDdictionary redis-cli ping
```
>如果返回 PONG，说明 Redis 运行正常！

---
## 管理 Redis 容器
停止 Redis（如果不想运行了）：
```shell
docker stop redis-server_NZDdictionary
```

重新启动 Redis（停止后再启动）：
```shell
docker start redis-server_NZDdictionary
```

删除 Redis 容器（如果不再需要）：
```shell
docker rm redis-server_NZDdictionary
```
查看 Redis 日志（调试用）：
```shell
docker logs redis-server_NZDdictionary
```
清空 Redis 
```shell
docker exec -it redis-server_NZDdictionary redis-cli FLUSHALL  # 清空 Redis
```