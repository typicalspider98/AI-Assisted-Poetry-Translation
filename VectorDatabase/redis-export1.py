import json
import redis
import numpy as np

# 连接 Redis
r = redis.Redis(host="localhost", port=6379, db=0)

def backup_redis(file_path="redis_backup_v1.0.json"):
    """
    备份 Redis 数据，将所有键值存入 JSON 文件
    """
    keys = r.keys("*")  # 获取所有 key
    backup_data = {}

    for key in keys:
        key_str = key.decode("utf-8")  # 转换 key 为字符串
        stored_embedding = r.get(key)  # 获取存储的二进制数据
        if stored_embedding:
            backup_data[key_str] = stored_embedding.hex()  # 转换为 hex 字符串存储

    # 保存到 JSON 文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(backup_data, f, indent=4)

    print(f"✅ Redis 数据已备份到 {file_path}")

# 执行备份
backup_redis()
